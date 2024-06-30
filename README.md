# guided-diffusion

This repository is forked from [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for identity conditioning


# Sampling from pre-trained models

To sample from these models, you can use the `image_sample.py`
To use identity embeddings, use `class_cond=True` and specify path to embeddings using `embeddings_path` flag. 
Here, we provide flags for sampling from from this models.

We will generate 100 samples with batch size 4. Feel free to change these values.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
MODEL_FLAGS=" --image_size 64 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --num_channels 192 --num_heads 3 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_new_attention_order True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule cosine --embedding_size 512"
python image_sample.py  --embeddings_path test_data/embeddings $MODEL_FLAGS --model_path models/saved_checkpoint.pt $SAMPLE_FLAGS
```

# Training models

We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are the parameters we used to trained identity-conditioned DDPM model on FFHQ dataset and their FaceNet embeddings. These paramaters are taken from the paper [Controllable Inversion of Black-Box Face Recognition Models via Diffusion](https://studios.disneyresearch.com/2023/10/02/controllable-inversion-of-black-box-face-recognition-models-via-diffusion/)
```
MODEL_FLAGS="--image_size 64 --num_channels 192 --num_heads 3 --num_res_blocks 3 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --resblock_updown True --use_fp16 True --use_new_attention_order True --embedding_size 512"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"
python scripts/image_train.py --data_dir path/to/images_and_embeddings $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```


