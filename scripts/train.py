import torch
from torch import nn
import torch.optim as optim
from dataset import dataset_import
from metrics import mean_absolute_percentage_error, RMSELoss
import sys
import numpy as np
import pretrainedmodels as models
import torch.nn.functional as F
import torchvision
from model import create_model


np.set_printoptions(threshold=sys.maxsize)
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
if DEVICE == "cuda":
    print("Running on x86 GPU")
else:
    print("Running on arm64 GPU")


# Define the NN model:
class NeuralNetwork(nn.Module):  # Inherits from the Module superclass
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30 * 30 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

        return output


def model_creation():
    model = NeuralNetwork().to(DEVICE)
    print(model)
    return model


def model_param_tweaking(model):
    weight_decay = 1e-5

    # Define a loss function and an optimiser
    loss_func = nn.MSELoss(reduction='mean')
    #loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), weight_decay=weight_decay)
    return loss_func, optimiser


def train_model(dataloader, model, loss_func, optimiser):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        '''print(f"{X}\ndim={X.shape[0]}:{X.shape[1]}")
        print(f"{y}\nlength={y.shape[0]}")
        print(f"{type(X[1])}")'''

        # Calculate prediction error:
        pred = abs(model(X))
        #print("pred: {}\n---------------\ntru: {}\n".format(pred, y))
        #print(f"type for y {type(y[0][0])}")
        #loss_value = loss_func(pred, y)
        loss_value = F.mse_loss(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        # Backpropagation
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()
        if batch % 100 == 0:
            mape = mean_absolute_percentage_error(y, pred)
            rmse = RMSELoss(y, pred)
            loss_value, current = loss_value.item(), batch * len(X)
            print(f"Train:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")


def test_model(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Perform evaluation
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            #test_loss += loss_func(pred, y).item()
            test_loss += F.mse_loss(pred, y).item()

    test_loss /= num_batches
    correct /= size
    mape = mean_absolute_percentage_error(y, pred)
    rmse = RMSELoss(y, pred)
    print(f"\nTest Error: \n rmse: {rmse:>0.4f}, mape: {mape:>0.4f}, Avg loss: {test_loss:>8f} \n")


def train(train_data_loader, model, optimiser):
    print('Training')
    size = len(train_data_loader.dataset)
    global train_itr  # A list that stores all the training iterations
    global train_loss_list  # A list that stores all the training loss values

    for batch, (X, y) in enumerate(train_data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = abs(model(X))  # Forward pass
        print("pred: {}\n---------------\ntru: {}\n".format(pred, y))

        loss_value = F.mse_loss(pred, y)
        if DEVICE == "mps":
            # mps framework supports float32 instead of 64 unlike cuda
            loss_value = loss_value.type(torch.float32)

        # Backpropagation
        optimiser.zero_grad()
        loss_value.backward()
        optimiser.step()

        if batch % 100 == 0:
            mape = mean_absolute_percentage_error(y, pred)
            rmse = RMSELoss(y, pred)
            loss_value, current = loss_value.item(), batch * len(X)
            print(f"Train:  loss: {loss_value:>7f}   [{current:>5d}/{size:>5d}]     rmse: {rmse:>0.4f}    mape: {mape:>0.4f}")
    #return train_loss_list


def main() -> object:
    train_dataloader, test_dataloader, test_loader = dataset_import()
    model = create_model().to(DEVICE)
    #model = model.train()
    print(model)
    # Get the model parameters: put each of the model's parameter value into a list
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=0.000245, momentum=0.9, weight_decay=0.0005)

    epochs = 20
    for i in range(epochs):
        print("Epoch {}\n__________________________________________".format(i + 1))
        #train_model(train_dataloader, model, loss_func, optimiser)
        train(train_dataloader, model, optimiser)
        #test_model(test_dataloader, model, loss_func)
    print("Completed")

    '''model_saved_state = save_model(model)
    model = load_model(model_saved_state)
    perform_prediction(model, test_dataloader)'''


if __name__ == '__main__':
    main()