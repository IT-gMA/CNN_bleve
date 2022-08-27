import torch
import torchvision.models as models
import pretrainedmodels
from torch import nn
from config import DROPOUT, DEVICE


def create_model():
    model = models.resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.dropout = nn.Dropout(DROPOUT)
    '''model.layer4[1].relu = nn.Softmax()
    model.layer4[2].relu = nn.Softmax()'''

    return model


if __name__ == '__main__':
    model = create_model().to(DEVICE)
    print(model)