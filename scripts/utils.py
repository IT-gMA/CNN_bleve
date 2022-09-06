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


def order_liquid_dist_features(dataset):
    '''
    The purpose of this function is to order the dataset with respect to distance to sensor in ascending order,
    therefore, each liquid type (Butane or Propane) will follow each other one by one i.e butane the propane the butane ...
    to ensure training, validation and testing sets have roughly equal liquid type representation
    '''
    ordered_dataset = []    # new ordered dataset
    butanes = dataset[0:22954]      # the first half of the dataset contains on butane
    propanes = dataset[22954:len(dataset)]      # the second half of the dataset contains on propane

    if ORDER_METHOD == 0:
        for a in range(0, 46, 1):
            # traverse the row by distance, i.e vist all liquids with distnace = 5 then all liquids with distance = 6 ... 50
            for i in range(a, len(propanes), 46):
                if i < len(
                        butanes):  # Since there are more propanes present, we'll arrange the first 499 butanes and 499 propanes first
                    # order: butane and then propane
                    ordered_dataset.append(butanes[i])
                    ordered_dataset.append(propanes[i])
                else:
                    # the last (500th) propane
                    ordered_dataset.append(propanes[i])
    else:
        # arrange a set of 5-50 for each liquid, followed by another set of 5-50 of a different liquid
        for i in range(0, len(propanes), 46):
            if i < len(butanes):
                for j in range(i, i + 46):
                    ordered_dataset.append(propanes[j])
                for j in range(i, i + 46):
                    ordered_dataset.append(butanes[j])
            else:
                for j in range(i, i + 46):
                    ordered_dataset.append(propanes[j])

    print(len(ordered_dataset))
    return ordered_dataset


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
            if DEVICE == "mps":
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
        # given each image has their corresponding output in the correct order
        if tensor == 0:
            dataset.append([img_list[i], image_names[i], output_list[i]])
        else:
            dataset.append([img_list[i], output_list[i]])

    # OPTIONAL to order the dataset
    if ORDER:
        dataset = order_liquid_dist_features(dataset)

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


def save_model(model, save_from_val=False, final=False, epoch=""):
    if not save_from_val and final:
        save_location = "{}/{}_final_model{}".format(SAVED_MODEL_DIR, SAVED_MODEL_NAME, SAVED_MODEL_FORMAT)
    elif save_from_val and not final:
        save_location = "{}/{}_best_model{}".format(SAVED_MODEL_DIR, SAVED_MODEL_NAME, SAVED_MODEL_FORMAT)
    else:
        save_location = "{}/{}_{}_model{}".format(SAVED_MODEL_DIR, SAVED_MODEL_NAME, epoch, SAVED_MODEL_FORMAT)
        print(f"Save model at {save_location}")
    torch.save(model.state_dict(), save_location)
    print("Pytorch model's state is saved to " + save_location)
    return save_location


def load_model(saved_location):
    ckpt = torch.load(saved_location, map_location=DEVICE)
    model = create_model(False)
    model.load_state_dict(ckpt)
    return model


def save_running_logs(info):
    print(info)

    log_file_name = SAVED_MODEL_NAME.replace(".pt", "")
    save_location = "{}/{}.txt".format(LOG_DIR, log_file_name)
    with open(save_location, 'a') as f:
        f.write(f"{info}\n")


def write_run_configs(n_train, n_val, n_test):
    run_config0 = "Time: {}\nDataset directory: {}\nModel: {}\nLearning rate: {}\n".format(datetime.datetime.now(), SAVE_IMG_DIR, MODEL_NAME, LEARNING_RATE)
    if SCHEDULED:
        run_config0 = f"{run_config0}\nMin learning rate: {MIN_LEARNING_RATE}\n"
    run_config1 = "Weight decay: {}\nPatience: {}\n Number of running epochs: {}\n".format(WEIGHT_DECAY, PATIENCE, NUM_EPOCHS)
    if USE_DROP_OUT:
        run_config1 = f"{run_config1}Drop out: {DROPOUT}\n"
    run_config2 = "MSE Reduction: {}\nTrain batch size: {}\nValidation batch size: {}\n".format(MSE_REDUCTION, BATCH_SIZE, VAL_BATCHSIZE)
    run_config3 = "Train-Val_Test ratio: {}-{}-{}\n".format(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    run_config4 = "     Number of train images: {}\n".format(n_train)
    run_config5 = "     Number of validation images: {}\n".format(n_val)
    run_config6 = "     Number of test images: {}\n".format(n_test)
    if ORDER:
        if ORDER_METHOD == 0:
            run_config6 = f"{run_config6}\nOrder method: liquid >> distance\n"
        else:
            run_config6 = f"{run_config6}\nOrder method: distance >> liquid\n"
    run_config7 = "\n"
    if RESCALE:
        run_config7 = "Rescale factor:\n    Width: {} pixels\n      Height: {} pixels\n".format(NUM_ROW, NUM_COLUMN)

    if EMPTY_CUDA_CACHE:
        run_config8 = "Empty cuda cache: True\n"
    else:
        run_config8 = "Empty cuda cache: False\n"

    config_write = f"{run_config0}{run_config1}{run_config2}{run_config3}{run_config4}{run_config5}{run_config6}{run_config7}{run_config8}\n"
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
    #output_txt = read_output_txt()
    #print(output_txt.reshape(-1, 1))
    #order_liquid_dist_features(None)

