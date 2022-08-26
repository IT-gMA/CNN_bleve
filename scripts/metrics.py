import torch
import torch.nn as nn


def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true)) * 100


def RMSELoss(y_true, y_pred):
    eps = 1e-6
    criterion = nn.MSELoss()
    #print(f"\n===============================\npredicted vs true: {y_pred} - {y_true}\n==============================\n")

    return torch.sqrt(criterion(y_true, y_pred) + eps)