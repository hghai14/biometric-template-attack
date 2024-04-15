"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_original_images = []


    # sampled_numbers = random.sample(range(0, 999), args.num_samples)
    # sampled_numbers.sort()
    # index = 0

    emb_size = 512

    emb_dir = os.path.join(args.test_data, "embeddings")
    emb_files = os.listdir(emb_dir)
    emb_files.sort()
    
    img_dir = os.path.join(args.test_data, "images")
    img_files = os.listdir(img_dir)
    img_files.sort()



    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}

        if args.class_cond:
            # NOTE: read embeddings from a file or something
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )

            """
                Read embeddings from a files
                read batch_size embeddings from a file and convert them to tensor
            """           
            start_idx = len(all_images) * args.batch_size

            files = [emb_files[i] for i in range(start_idx, start_idx+args.batch_size)]
            files = [os.path.join(emb_dir, f) for f in files]
            classes = th.tensor(np.array([np.squeeze(np.load(f)) for f in files]), device=dist_util.dev())


            files = [img_files[i] for i in range(start_idx, start_idx+args.batch_size)]
            files = [os.path.join(img_dir, f) for f in files]
            original_images = th.tensor(np.array([np.array(Image.open(f)) for f in files]), device=dist_util.dev())
            

            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            gathered_original_images = [
                th.zeros_like(original_images) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_original_images, original_images)
            all_original_images.extend([original_images.cpu().numpy() for original_images in gathered_original_images])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        original_images_arr = np.concatenate(all_original_images, axis=0)
        original_images_arr = original_images_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr, original_images_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        test_data="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
