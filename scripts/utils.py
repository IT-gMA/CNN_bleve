import os

import numpy as np
import random
import cv2
import glob, sys
from natsort import natsorted, ns
import torch, torchvision

OUTPUT_FILE = '../data/outputs/outputs.txt'
IMG_DIR = '../data'
np.set_printoptions(threshold=sys.maxsize)
NUM_ROW = 30
NUM_COLUMN = 30


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well. I.e collect/gather training and validation loss
    values
    """
    return tuple(zip(*batch))  # To accumulate


def read_output_txt():
    output_list = []
    with open(OUTPUT_FILE, 'r') as f:
        outputs = f.readlines()
        for output in outputs:
            output = float(output[:-1])  # Detach "new line" from each read line
            #print(output)
            output_list.append(output)

    output_list = np.float32(output_list)
    output_list = torch.from_numpy(np.asarray(output_list))
    return output_list


def read_img():
    transform = torchvision.transforms.ToTensor()
    dataset_img = []
    img_paths = glob.glob("{}/*.png".format(IMG_DIR))
    img_paths = natsorted(img_paths, key=lambda y: y.lower())   # Sort the images in alphanumerical order
    for img in img_paths:
        #print(f"read {img}")
        bleve_img = cv2.imread(img)
        # Resize each image to its intended size after converted from tabular data form
        bleve_img = cv2.resize(bleve_img, (NUM_ROW, NUM_COLUMN), interpolation=cv2.INTER_AREA)
        #bleve_img = cv2.cvtColor(bleve_img, cv2.COLOR_RGB2GRAY)
        bleve_img = np.float32(bleve_img)
        bleve_img = transform(bleve_img)
        #print(bleve_img.shape)
        dataset_img.append(bleve_img)

    return dataset_img


def create_raw_dataset():
    dataset = []
    output_list = read_output_txt()
    img_list = read_img()
    check_data_vs_output_quantity(img_list, output_list)

    for i in range (0, len(output_list)):
        dataset.append([img_list[i], output_list[i]])

    random.shuffle(dataset)
    return dataset


def check_data_vs_output_quantity(img_list, output_list):
    num_img = len(img_list)
    num_output = len(output_list)
    if num_img < num_output:
        Exception(f"Missing {num_output - num_img} images in the dataset")
    elif num_img > num_output:
        Exception(f"Missing {num_img - num_output} output vales in file {OUTPUT_FILE}")


if __name__ == '__main__':
    dataset = create_raw_dataset()

    '''saved_dir = '../data/tru_sized_imgs'
    os.chdir(saved_dir)
    for i in range(0, 4):
        print(dataset[i][0].dtype)
        cv2.imshow(f"{dataset[i][1]}", dataset[i][0])
        cv2.imwrite(f"{dataset[i][1]}.png", dataset[i][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

