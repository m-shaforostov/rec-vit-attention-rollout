import torch.nn as nn


class PETAlexNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(PETAlexNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.pretrained.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        x = self.pretrained(x)
        return x


class PETResNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(PETResNet, self).__init__()
        self.pretrained = my_pretrained_model
        num_ftrs = my_pretrained_model.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, 37)

    def forward(self, x):
        x = self.pretrained(x)
        return x


class PETVGG(nn.Module):
    def __init__(self, my_pretrained_model):
        super(PETVGG, self).__init__()
        self.pretrained = my_pretrained_model
        self.pretrained.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        x = self.pretrained(x)
        return x

