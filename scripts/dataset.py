import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 256
NUM_WORKERS = 0


def retrieve_dataset():
    """Dataset retrived is in the form (Torch tensor):
        [0]: image
        [1]: corresponding output value"""
    dataset = utils.create_raw_dataset()

    num_train = int(len(dataset) * TRAIN_RATIO)
    num_val = int(len(dataset) * VAL_RATIO)
    num_test = len(dataset) - num_train - num_val

    train_set = dataset[0:num_train - 1]
    val_set = dataset[num_train:num_train + num_val - 1]
    test_set = dataset[num_train + num_val: len(dataset) - 1]

    return train_set, val_set, test_set


def dataset_import():
    train_dataset, valid_dataset, test_dataset = retrieve_dataset()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    for X, y in train_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = dataset_import()
    print(train_loader.dataset)
