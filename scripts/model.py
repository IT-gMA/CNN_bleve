import torch
import torchvision.models as models
import pretrainedmodels
from torch import nn


def create_model():
    model = models.resnet101(pretrained=True, progres=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    return model
