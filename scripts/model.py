import torch
import torchvision.models as models
import pretrainedmodels
from torch import nn
from config import DROPOUT, DEVICE


def create_model():
    model = models.resnet152(pretrained=True)
    model = models.googlenet(pretrained=True, progress=True)
    '''model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)'''

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model.dropout = nn.Dropout(DROPOUT)

    return model


if __name__ == '__main__':
    model = create_model().to(DEVICE)
    print(model)