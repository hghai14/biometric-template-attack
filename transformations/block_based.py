import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import random
from sys import argv

args = argv[1:]

# params
# read_folder = "ffhq256_pp/"
# block_size = 40
# maximum_pixel_offset = 5
# blur_size = 5

read_folder = args[0]
block_size = args[1]
maximum_pixel_offset = args[2]
blur_size = args[3]

key = 1234
seed = 0

# create save folder
save_folder = "warped_dataset/{}_{}_{}/".format(block_size, maximum_pixel_offset, blur_size)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# create sub folder for train and test
if not os.path.exists(save_folder + "/train/images"):
    os.makedirs(save_folder + "/train/images")
if not os.path.exists(save_folder + "/test/images"):
    os.makedirs(save_folder + "/test/images")


#functions
def get_random_index(start, end, n, seed = None):
    if seed == None:
        random.seed()
    else:
        random.seed(seed)
    return sorted(random.sample(range(start, end), n))
    
def block_based_warping(image, block_size, key, maximum_pixel_offset, k_size = 15):
    # Create regular grid of blocks
    rows, cols = image.shape[:2]
    block_rows, block_cols = rows // block_size, cols // block_size

    # Initialize the warped output image
    warped_image = np.zeros_like(image)

    # Set random seed based on the key
    np.random.seed(key)

    for block_row in range(block_rows):
        for block_col in range(block_cols):
            # Get block indices
            start_row = block_row * block_size
            end_row = start_row + block_size
            start_col = block_col * block_size
            end_col = start_col + block_size

            # Calculate random pixel offsets
            row_offset = np.random.randint(-maximum_pixel_offset, maximum_pixel_offset)
            col_offset = np.random.randint(-maximum_pixel_offset, maximum_pixel_offset)

            # Calculate the warped grid for the block
            warped_start_row = max(0, start_row + row_offset)
            warped_end_row = min(rows, end_row + row_offset)
            warped_start_col = max(0, start_col + col_offset)
            warped_end_col = min(cols, end_col + col_offset)

            # Perform spline interpolation to warp the block
            warped_block = cv2.resize(image[warped_start_row:warped_end_row, warped_start_col:warped_end_col],
                                      (block_size, block_size))

            # Assign the warped block to the corresponding region in the output image
            warped_image[start_row:end_row, start_col:end_col] = warped_block

    # Apply Gaussian blur to smoothen the output image
    kernel_size = (k_size, k_size)  # Increase the kernel size for more smoothing
    smoothed_image = cv2.GaussianBlur(warped_image, kernel_size, 0)

    return smoothed_image


# sample 1000 images for train and test
train_samples = get_random_index(0,68999,1000,seed)
test_samples = range(69000, 70000)

# warp images
for i in train_samples:
    try:
        image = cv2.imread(read_folder + "train/images/{:05d}.png".format(i), cv2.IMREAD_COLOR)
        warped_image = block_based_warping(image, block_size, key, maximum_pixel_offset, blur_size)
        cv2.imwrite(save_folder + "train/images/{:05d}.png".format(i), warped_image)
    except:
        print("train image does not exist: ", i)
        continue

for i in test_samples:
    try:
        image = cv2.imread(read_folder + "test/images/{:05}.png".format(i), cv2.IMREAD_COLOR)
        warped_image = block_based_warping(image, block_size, key, maximum_pixel_offset, blur_size)
        cv2.imwrite(save_folder + "test/images/{:05}.png".format(i), warped_image)
    except:
        print("test image does not exist: ", i)
        continue