import os
import datetime
import numpy as np
import random
import cv2
import glob, sys
from natsort import natsorted, ns
import torch, torchvision
from config import *
from model import create_model

np.set_printoptions(threshold=sys.maxsize)


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well. I.e collect/gather training and validation loss
    values
    """
    return tuple(zip(*batch))  # To accumulate


def read_output_txt(tensor=0):
    output_list = []

    output_file_path = OUTPUT_FILE
    if tensor == 1:
        output_file_path = SAVE_OUTPUT_DIR

    with open(output_file_path, 'r') as f:
        outputs = f.readlines()
        for output in outputs:
            output = float(output[:-1])
            '''if tensor == 1:
                print(f"reading output: {output}")'''
            output_list.append(output)

    if tensor == 1:
        output_list = np.float32(output_list)
        output_list = torch.from_numpy(np.asarray(output_list))

    return output_list


def read_img(tensor=0):
    dataset_img = []
    image_names = []

    img_dir = IMG_DIR
    if tensor == 1:
        img_dir = SAVE_IMG_DIR
        transform = torchvision.transforms.ToTensor()

    img_paths = glob.glob("{}/*{}".format(img_dir, FILE_EXTENSION))
    img_paths = natsorted(img_paths, key=lambda y: y.lower())   # Sort the images in alphanumerical order

    for img in img_paths:
        img_name = get_gas_name(img)
        image_names.append(img_name)
        print(f"reading {img_name}")

        bleve_img = cv2.imread(img)
        # Resize each image to its intended size after converted from tabular data form
        #bleve_img = cv2.resize(bleve_img, (NUM_ROW, NUM_COLUMN), interpolation=cv2.INTER_AREA)

        if tensor == 1:
            if RESCALE:
                bleve_img = cv2.resize(bleve_img, (NUM_ROW, NUM_COLUMN), interpolation=cv2.INTER_AREA)

            #bleve_img = cv2.cvtColor(bleve_img, cv2.COLOR_RGB2GRAY)
            bleve_img = np.float32(bleve_img)
            bleve_img = transform(bleve_img)
            #print(bleve_img.shape)

        dataset_img.append(bleve_img)

    return dataset_img, image_names


def get_gas_name(file_name):
    file_name = file_name.replace('_image.png', '')
    file_name = file_name.replace('_', '')
    file_name = file_name.replace(f'{IMG_DIR}/', '')
    return file_name


def create_raw_dataset(tensor=0):
    if tensor == 0:
        print("Create permanent shuffled dataset at {}.".format(SAVE_IMG_DIR))
    elif tensor == 1:
        print("Convert data in {} into tensor form for model training.".format(SAVE_IMG_DIR))
    if tensor != 0 or tensor != 1:
        Exception(f"Invalid value for tensor = {tensor} -\n"
                  f"0: not convert data to tensors.\n"
                  f"1: convert data to tensors.")

    dataset = []
    output_list = read_output_txt(tensor)
    img_list, image_names = read_img(tensor)
    check_data_vs_output_quantity(img_list, output_list)

    for i in range(0, len(output_list)):
        if tensor == 0:
            dataset.append([img_list[i], image_names[i], output_list[i]])
        else:
            dataset.append([img_list[i], output_list[i]])

    if tensor == 0:
        random.shuffle(dataset)

    return dataset


def check_data_vs_output_quantity(img_list, output_list):
    num_img = len(img_list)
    num_output = len(output_list)
    if num_img < num_output:
        Exception(f"Missing {num_output - num_img} images in the dataset")
    elif num_img > num_output:
        Exception(f"Missing {num_img - num_output} output vales in file {OUTPUT_FILE}")


def save_model(model, save_from_val=False):
    if not save_from_val:
        save_location = "{}/{}_final_model{}".format(SAVED_MODEL_DIR, SAVED_MODEL_NAME, SAVED_MODEL_FORMAT)
    else:
        save_location = "{}/{}_best_model{}".format(SAVED_MODEL_DIR, SAVED_MODEL_NAME, SAVED_MODEL_FORMAT)
    torch.save(model.state_dict(), save_location)
    print("Pytorch model's state is saved to " + save_location)
    return save_location


def load_model(saved_location):
    model = create_model()
    model.load_state_dict(torch.load(saved_location))
    return model


def save_running_logs(info):
    print(info)

    log_file_name = SAVED_MODEL_NAME.replace(".pt", "")
    save_location = "{}/{}.txt".format(LOG_DIR, log_file_name)
    with open(save_location, 'a') as f:
        f.write(f"{info}\n")


def write_run_configs(n_train, n_val, n_test):
    run_config0 = "Time: {}\nDataset directory: {}\nModel: {}\nLearning rate: {}\n".format(datetime.datetime.now(), SAVE_IMG_DIR, MODEL_NAME, LEARNING_RATE)
    run_config1 = "Weight decay: {}\nDrop out: {}\n Patience: {}\n Number of running epochs: {}\n".format(WEIGHT_DECAY, DROPOUT, PATIENCE, NUM_EPOCHS)
    run_config2 = "Train batch size: {}\nValidation batch size: {}\n".format(BATCH_SIZE, VAL_BATCHSIZE)
    run_config3 = "Train-Val_Test ratio: {}-{}-{}\n".format(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    run_config4 = "     Number of train images: {}\n".format(n_train)
    run_config5 = "     Number of validation images: {}\n".format(n_val)
    run_config6 = "     Number of test images: {}\n".format(n_test)
    run_config7 = ""
    if RESCALE:
        run_config7 = "Rescale factor:\n    Width: {} pixels\n      Height: {} pixels\n".format(NUM_ROW, NUM_COLUMN)

    config_write = f"{run_config0}{run_config1}{run_config2}{run_config3}{run_config4}{run_config5}{run_config6}{run_config7}\n"
    save_running_logs(config_write)


if __name__ == '__main__':
    #dataset = create_raw_dataset()

    '''saved_dir = '../data/tru_sized_imgs'
    os.chdir(saved_dir)
    for i in range(0, 4):
        print(dataset[i][0].dtype)
        cv2.imshow(f"{dataset[i][1]}", dataset[i][0])
        cv2.imwrite(f"{dataset[i][1]}.png", dataset[i][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
    output_txt = read_output_txt()
    print(output_txt.reshape(-1, 1))

