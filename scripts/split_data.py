import cv2
import os
from functools import reduce
from progress.bar import IncrementalBar
import numpy as np
import sys
from config import ORDER_METHOD, SEED_RANGE, DEFAULT_DATASET_DIR, STORED_DIR, OG_DATASET_PATH, FILE_EXTENSION
import random
from utils import run_dir_exist, create_raw_dataset

np.set_printoptions(threshold=sys.maxsize)


def fill_sub_dir(sub_data_dir, output_dir, seed):
    dataset_instance = create_raw_dataset(tensor=0, seed=seed)
    i = 1
    output_list = []
    for data in dataset_instance:
        img = data[0]
        output = data[1]

        cv2.imwrite(f"{sub_data_dir}/img_{i}.{FILE_EXTENSION}", img)
        output_list.append(output)
        i += 1

    with open(f"{output_dir}/outputs.txt", 'a') as f:
        i = 1
        for instance in output_list:
            if i < len(output_list):
                f.writelines(f"{instance}\n")
            else:
                f.writelines(f"{instance}")
            i += 1


def create_sub_dirs(data_dir, mode=0o777):
    for seed in range(0, SEED_RANGE):
        sub_data_dir = os.path.join(data_dir, f"seed_{seed}")
        if not run_dir_exist(sub_data_dir):
            print("creating sub dir {}".format(sub_data_dir))
            os.mkdir(sub_data_dir, mode)

            output_dir = os.path.join(sub_data_dir, "outputs")
            os.mkdir(output_dir, mode)
            fill_sub_dir(sub_data_dir, output_dir, seed)


def init_seeded_dataset_dir():
    inner_dir = OG_DATASET_PATH.replace(f"{DEFAULT_DATASET_DIR}/", "")
    data_dir = os.path.join(STORED_DIR, inner_dir)

    if not run_dir_exist(data_dir):     # check for empty directory
        mode = 0o777
        #print("Create main dataset location at {}".format(data_dir))
        os.mkdir(data_dir, mode)

    create_sub_dirs(data_dir)


if __name__ == '__main__':
    init_seeded_dataset_dir()
