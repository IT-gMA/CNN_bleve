import torch
import torchvision.models as models
import pretrainedmodels
from torch import nn
from config import DROPOUT


def create_model():
    model = models.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.dropout = nn.Dropout(DROPOUT)

    return model
