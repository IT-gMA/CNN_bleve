import torch
from torch import nn
import torch.optim as optim
from dataset import dataset_import
from metrics import mean_absolute_percentage_error, RMSELoss
import sys
import numpy as np
from config import LEARNING_RATE, MIN_LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DEVICE, MODEL_NAME
import torch.nn.functional as F
import torchvision
from model import create_model


np.set_printoptions(threshold=sys.maxsize)


def model_param_tweaking(model):
    loss_func = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                           factor=0.1, patience=16, threshold=0.0001,
                                                           threshold_mode='abs')
    return loss_func, optimiser, scheduler


def test(dataloader, model, loss_func):
    print("Final testing")
    size = len(dataloader.dataset)

    # Start final testing
    model.eval()
    for batch, (X, y) in enumerate(dataloader):
        # Forward pass
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = get_predictions(X, model)

        loss_value = loss_func(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        mape = mean_absolute_percentage_error(y, pred)
        rmse = RMSELoss(y, pred)
        loss_value, current = loss_value.item(), batch * len(X)
        print("pred: {}\ntru: {}".format(pred, y))
        print(
            f"Test:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")


def train(train_dataloader, model, loss_func, optimiser):
    size = len(train_dataloader.dataset)
    total_loss = 0

    # Training
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass
        pred = get_predictions(X, model)
        #print("pred: {}\n---------------\ntru: {}\n".format(pred, y))

        loss_value = loss_func(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        # Backpropagation
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()
        total_loss += loss_value

        if batch % 100 == 0:
            mape = mean_absolute_percentage_error(y, pred)
            rmse = RMSELoss(y, pred)
            loss_value, current = loss_value.item(), batch * len(X)
            #print("pred: {}\ntru: {}".format(pred, y))
            print(f"Train:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}\n")

    return total_loss


def validation(val_dataloader, model, loss_func):
    size = len(val_dataloader.dataset)

    # Start model evaluation
    model.eval()
    for batch, (X, y) in enumerate(val_dataloader):
        # Forward pass
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = get_predictions(X, model)

        loss_value = loss_func(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        mape = mean_absolute_percentage_error(y, pred)
        rmse = RMSELoss(y, pred)
        loss_value, current = loss_value.item(), batch * len(X)
        print("pred: {}\ntru: {}".format(pred, y))
        print(f"Validation:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")


def get_predictions(input, model):
    if MODEL_NAME == "inceptionv3":
        predictions = model(input)[0].squeeze()
    else:
        predictions = model(input).squeeze()

    return predictions


def main() -> object:
    train_dataloader, validation_dataloader, test_loader = dataset_import()
    model = create_model().to(DEVICE)
    print(model)
    loss_func, optimiser, lr_scheduler = model_param_tweaking(model)

    epochs = NUM_EPOCHS
    for i in range(epochs):
        print("Epoch {}\n_________________________________________________________________".format(i + 1))
        train_loss = train(train_dataloader, model, loss_func, optimiser)
        lr_scheduler.step(train_loss)
        if i % 5 == 0 and i > 1:
            print("---------------------------------VALIDATION AT EPOCH {}-----------------------------------".format(i+1))
            validation(validation_dataloader, model, loss_func)
            print("------------------------------------END OF VALIDATION----------------------------------------\n")

        print("_________________________________________________________________________________________")

    print("Training complete")
    test(test_loader, model, loss_func)

    '''model_saved_state = save_model(model)
    model = load_model(model_saved_state)
    perform_prediction(model, test_dataloader)'''


if __name__ == '__main__':
    main()