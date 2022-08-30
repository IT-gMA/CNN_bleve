import torch
import torchvision.models as models
import pretrainedmodels
from torch import nn
from config import DROPOUT, DEVICE, MODEL_NAME


def create_model():
    if MODEL_NAME == "resnet50":
        # Resnet50
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features

        #model.fc = nn.Linear(num_features, 1)
        model.fc = nn.Sequential(nn.Linear(num_features, 256),
                      nn.LeakyReLU(),
                      nn.Linear(256, 32),
                      nn.LeakyReLU(),
                      nn.Linear(32, 1))

    elif MODEL_NAME == "efficientnet_b7":
        # Efficientnet_b7
        model = models.efficientnet_b7(pretrained=True)
        num_features = model.classifier[1].in_features
        #model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.classifier = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(num_features, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32, 1))

    elif MODEL_NAME == "inceptionv3":
        # Inceptionv3
        model = models.inception_v3(pretrained=True)

        num_features = model.fc.in_features
        #model.fc = nn.Linear(num_features, 1)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
        model.fc = nn.Sequential(nn.Linear(num_features, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32, 1))
    else:
        Exception(f"{MODEL_NAME} is an invalid model's name")

    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.dropout = nn.Dropout(DROPOUT)

    return model


if __name__ == '__main__':
    model = create_model().to(DEVICE)
    print(model)