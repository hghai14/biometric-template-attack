import os
import random

random.seed(0) # for reproducibility
# for folders in FEI, LFW, etc. select one image from all sub-folders

path = '/home/hp/Desktop/sem10/mtp-2/datasets/LFW'
copy_path = '/home/hp/Desktop/sem10/mtp-2/LFW'

folders = os.listdir(path)
folders.sort()
images = []
for folder in folders:
    files = os.listdir(path + '/' + folder)
    image = random.choice(os.listdir(path + '/' + folder))
    os.system('cp ' + path + '/' + folder + '/' + image + ' ' + copy_path + '/' + folder + '.jpg')
