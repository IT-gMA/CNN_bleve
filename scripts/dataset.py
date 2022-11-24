import torch
from config import TRAIN_RATIO, VAL_RATIO, TRAIN_AUG, BATCH_SIZE, NUM_WORKERS, VAL_BATCHSIZE, SHUFFLE_TRAIN, SHUFFLE_VAL, PROGRESS_SLEEP_TIME
import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from progress.bar import FillingSquaresBar
import random
from time import sleep


def retrieve_dataset(seed):
    """Dataset retrived is in the form (Torch tensor):
        [0]: image
        [1]: corresponding output value"""
    dataset = utils.create_raw_dataset(tensor=1, seed=seed)

    num_train = int(len(dataset) * TRAIN_RATIO)
    num_val = int(len(dataset) * VAL_RATIO)
    num_test = len(dataset) - num_train - num_val

    train_set = dataset[0:num_train]
    val_set = dataset[num_train:num_train + num_val]
    test_set = dataset[num_train + num_val: len(dataset)]

    '''if TRAIN_AUG:
        train_set = transform_train_set(train_set)
        print(f"{len(train_set)} train images")'''
    '''for data in train_set:
        print(data)'''

    return train_set, val_set, test_set, num_train, num_val, num_test


def transform_train_set(train_set):
    aug_train_set = []
    transform_list = get_transformations()

    with FillingSquaresBar('Applying augmentation...', max=len(train_set)) as bar:
        #sleep(PROGRESS_SLEEP_TIME)
        for data in train_set:
            img = data[0]
            output = data[1]
            aug_train_set.append([img, output])

            for transformations in transform_list:
                aug_img = transformations(img)
                aug_train_set.append([aug_img, output])
                # Add image noise

            aug_img = add_noise(img, noise_factor=0.25)
            aug_train_set.append([aug_img, output])
            bar.next()

    return aug_train_set


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

    return [t0, t1, t2, t4, t5]


def probabilistic_transformation(img_batch):
    transformation_list = get_transformations()
    '''augmentation = random.choice(transformation_list)
    if augmentation == -1:
        # print("Add noise")
        return add_noise(img_batch, noise_factor=0.25)
    else:
        # print(augmentation)
        return augmentation(img_batch)'''

    for i in range(len(img_batch)):
        augmentation = random.choice(transformation_list)
        if augmentation == -1:
            # print("Add noise")
            img_batch[i] = add_noise(img_batch[i], noise_factor=0.25)
        else:
            # print(augmentation)
            img_batch[i] = augmentation(img_batch[i])


def getTransformation_ext():
    #t1 = Compose([transforms.RandomPerspective(distortion_scale=0.40, p=0.60)])
    #transforms.RandomRotation(degrees=(0, 35))])
    '''t2 = Compose([transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.05, hue=0.125),
                  transforms.RandomInvert(p=0.35)])'''
    t2 = Compose([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                  transforms.RandomAutocontrast()])

    img_transform = [t2]
    return img_transform


def dataset_import(inference=False, seed=0, model=None):
    train_dataset, valid_dataset, test_dataset, num_train, num_val, num_test = retrieve_dataset(seed)

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


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = dataset_import()
    '''for X, y in train_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break'''
    #print(train_loader.dataset)
