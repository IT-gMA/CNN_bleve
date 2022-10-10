import torch
from config import TRAIN_RATIO, VAL_RATIO, TRAIN_AUG, BATCH_SIZE, NUM_WORKERS, VAL_BATCHSIZE, SHUFFLE_TRAIN, SHUFFLE_VAL
import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose


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

    if TRAIN_AUG:
        train_set = transform_train_set(train_set)
        print(f"{len(train_set)} train images")
    '''for data in train_set:
        print(data)'''

    return train_set, val_set, test_set, num_train, num_val, num_test


def transform_train_set(train_set):
    aug_train_set = []
    transform_list = get_transformations()
    i = 1
    for data in train_set:
        print(f"applying augmentation for image {i}")
        img = data[0]
        output = data[1]
        aug_train_set.append([img, output])
        for transformations in transform_list:
            aug_img = transformations(img)
            aug_train_set.append([aug_img, output])

        # Add image noise
        aug_img = add_noise(img, noise_factor=0.25)
        aug_train_set.append([aug_img, output])
        i += 1
    return aug_train_set


def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


def get_transformations():
    t1 = Compose([transforms.RandomPerspective(distortion_scale=0.40, p=0.60),
                  transforms.RandomRotation(degrees=(0, 35))])
    t2 = Compose([transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.05, hue=0.125),
                  transforms.RandomInvert(p=0.35)])
    t3 = Compose([transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)),
                  transforms.ElasticTransform(alpha=100.0)])

    img_transform = [t1, t2, t3]
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
