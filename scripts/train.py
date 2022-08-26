import torch
from torch import nn
import torch.optim as optim
from dataset import dataset_import
from metrics import mean_absolute_percentage_error, RMSELoss
import sys
import numpy as np
from config import LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DEVICE
import torch.nn.functional as F
import torchvision
from model import create_model


np.set_printoptions(threshold=sys.maxsize)


def model_param_tweaking(model):
    loss_func = nn.MSELoss(reduction='mean')
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return loss_func, optimiser


def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Perform evaluation
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            #test_loss += loss_func(pred, y).item()
            test_loss += loss_func(pred.squeeze(), y).item()

    test_loss /= num_batches
    mape = mean_absolute_percentage_error(y, pred.squeeze())
    rmse = RMSELoss(y, pred)
    print(f"\nTest Error: \n rmse: {rmse:>0.4f}, mape: {mape:>0.4f}, Avg loss: {test_loss:>8f} \n")


def train(train_dataloader, model, loss_func, optimiser):
    print('Training')
    size = len(train_dataloader.dataset)
    global train_itr  # A list that stores all the training iterations
    global train_loss_list  # A list that stores all the training loss values

    # Training
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass
        pred = abs(model(X))
        #print("pred: {}\n---------------\ntru: {}\n".format(pred, y))

        loss_value = loss_func(pred.squeeze(), y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        # Backpropagation
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()

        if batch % 100 == 0:
            mape = mean_absolute_percentage_error(y, pred.squeeze())
            rmse = RMSELoss(y, pred)
            loss_value, current = loss_value.item(), batch * len(X)
            print(f"Train:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")


def validation(val_dataloader, model, loss_func):
    size = len(val_dataloader.dataset)
    for batch, (X, y) in enumerate(val_dataloader):
        # Forward pass
        X, y = X.to(DEVICE), y.to(DEVICE)
        model.eval()
        pred = abs(model(X))

        loss_value = loss_func(pred.squeeze(), y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        mape = mean_absolute_percentage_error(y, pred.squeeze())
        rmse = RMSELoss(y, pred)
        loss_value, current = loss_value.item(), batch * len(X)
        print(f"Validation:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")


def main() -> object:
    train_dataloader, validation_dataloader, test_loader = dataset_import()
    model = create_model().to(DEVICE)
    model = model.train()
    print(model)
    # Get the model parameters: put each of the model's parameter value into a list
    params = [p for p in model.parameters() if p.requires_grad]
    loss_func, optimiser = model_param_tweaking(model)
    optimiser = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    epochs = NUM_EPOCHS
    for i in range(epochs):
        print("Epoch {}\n__________________________________________".format(i + 1))
        #train_model(train_dataloader, model, loss_func, optimiser)
        train(train_dataloader, model, loss_func, optimiser)
        if i % 5 == 0:
            validation(validation_dataloader, model, loss_func)
    print("Completed")

    '''model_saved_state = save_model(model)
    model = load_model(model_saved_state)
    perform_prediction(model, test_dataloader)'''


if __name__ == '__main__':
    main()