import os
import torch
from config import TRAIN_RATIO, VAL_RATIO, TRAIN_AUG, BATCH_SIZE, NUM_WORKERS, VAL_BATCHSIZE, SHUFFLE_TRAIN, SHUFFLE_VAL, FILE_EXTENSION, RESCALE, NUM_COLUMN, NUM_ROW
import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from progress.bar import FillingSquaresBar
import random
import cv2 as cv
import glob
import numpy as np
import sys
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)


class BleveDataset(Dataset):
    def __init__(self, output_file, dataset_dir, trasnform=None):
        # data loading
        self.outputs = utils.read_output_txt(tensor=1, output_path=output_file)
        self.dataset_dir = dataset_dir
        self.transform = trasnform
        self.image_paths = glob.glob("{}/*{}".format(self.dataset_dir, FILE_EXTENSION.replace('.', '')))

    def __getitem__(self, idx):
        img_list = []
        output_list = []
        dataset = []
        # capture the image name and the full image path
        image_names = self.image_paths[idx]
        with FillingSquaresBar('Reading images...', max=len(image_names)) as bar:
            for img_name in image_names:
                img = cv.imread(img_name)
                pre_transform = Compose([transforms.ToTensor()])
                if RESCALE:
                    pre_transform = Compose([transforms.ToTensor(), transforms.Resize(size=[NUM_ROW, NUM_COLUMN])])
                img = pre_transform(img)
                if self.transform:
                    img = self.transform(img)
                img_list.append(img)
                bar.next()

        all_outputs = self.outputs[idx]
        for output in all_outputs:
            output_list.append(output)

        for i in range(len(img_list)):
            dataset.append([img_list[i], output_list[i]])

        return dataset

    def __len__(self):
        # allow for len(dataset) call
        return len(self.image_paths)


def retrieve_dataset(output_file, dataset_dir, transform):
    """Dataset retrived is in the form (Torch tensor):
        [0]: image
        [1]: corresponding output value"""
    dataset = BleveDataset(output_file=output_file,
                           dataset_dir=dataset_dir,
                           trasnform=None)

    num_train = int(len(dataset) * TRAIN_RATIO)
    num_val = int(len(dataset) * VAL_RATIO)
    num_test = len(dataset) - num_train - num_val

    print(f"Train set: {num_train} images")
    train_set = dataset[0:num_train]
    print(f"Validation set: {num_val} images")
    val_set = dataset[num_train:num_train + num_val]
    print(f"Test set: {num_test} images")
    test_set = dataset[num_train + num_val: len(dataset)]

    return train_set, val_set, test_set, num_train, num_val, num_test


def dataset_import(inference=False, seed=0, model=None):
    dataset_dir, output_file = utils.get_saved_dataset_path(seed)
    train_dataset, valid_dataset, test_dataset, num_train, num_val, num_test = retrieve_dataset(output_file=output_file,
                                                                                                dataset_dir=dataset_dir,
                                                                                                transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=VAL_BATCHSIZE,
        shuffle=SHUFFLE_VAL,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    transform_list = get_transformations()
    if not inference:
        utils.write_run_configs(num_train, num_val, num_test, seed=seed, model=model, transformations=transform_list)
        return train_loader, validation_loader, test_loader
    else:
        return test_loader


def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


def get_transformations():
    t0 = Compose([transforms.RandomRotation(degrees=(0, 0))])      # no transformation
    t1 = Compose([transforms.RandomPerspective(distortion_scale=0.30, p=1.0)])
    t2 = Compose([transforms.RandomAdjustSharpness(sharpness_factor=5, p=1.0),
                 transforms.RandomPerspective(distortion_scale=0.10, p=0.5)])
    t3 = Compose([transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.05, hue=0.125)])
    t4 = Compose([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                  transforms.RandomAutocontrast()])
    t5 = -1     # add noise transformation

    img_transform = [t0, t1, t2, t3, t4, t5]
    return img_transform


def probabilistic_transformation(img_batch):
    transformation_list = get_transformations()

    for i in range(len(img_batch)):
        augmentation = random.choice(transformation_list)
        if augmentation == -1:
            #print("Add noise")
            img_batch[i] = add_noise(img_batch[i], noise_factor=0.25)
        else:
            #print(augmentation)
            img_batch[i] = augmentation(img_batch[i])

        '''to_pil_trans = Compose([transforms.ToPILImage()])
        aug_img = to_pil_trans(img_batch[i]) #img_batch[i].detach().cpu().numpy()
        aug_img = np.array(aug_img)
        aug_img = cv.cvtColor(aug_img, cv.COLOR_RGB2BGR)
        cv.imshow("image", aug_img)
        cv.waitKey(0)'''


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = dataset_import(seed=0)
    #probabilistic_transformation(None)
