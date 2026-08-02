import torch
from torch import nn
import torchvision


class CIFAR10_AlexNet(nn.Module):
    # used for generating AEs
    def __init__(self):
        super(CIFAR10_AlexNet, self).__init__()

        pretrained_model = torchvision.models.alexnet(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample((160, 160))
        self.pad = nn.ZeroPad2d(32)
        self.pretrained = pretrained_model

        self.pretrained.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x


class CIFAR10_ResNet(nn.Module):
    # used for generating AEs
    def __init__(self):
        super(CIFAR10_ResNet, self).__init__()

        pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample((160, 160))
        self.pad = nn.ZeroPad2d(32)
        self.pretrained = pretrained_model

        num_ftrs = pretrained_model.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x


class CIFAR10_VGG11(nn.Module):
    # used for generating AEs
    def __init__(self):
        super(CIFAR10_VGG11, self).__init__()

        pretrained_model = torchvision.models.vgg11_bn(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.up = nn.Upsample((160, 160))
        self.pad = nn.ZeroPad2d(32)
        self.pretrained = pretrained_model
        self.pretrained.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.pad(x)
        x = self.pretrained(x)
        return x


class CIFAR10_basic(nn.Module):
    # used for cross-validating AEs
    def __init__(self):
        super(CIFAR10_basic, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.activation = torch.Tensor()
        self.classifier = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.2),

                                        nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.3),

                                        nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate'),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Dropout(0.4),

                                        nn.Flatten(),
                                        nn.BatchNorm1d(2048),
                                        nn.Linear(2048, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 10)
                                        )

    def forward(self, x):
        x = self.classifier(x)
        return x
