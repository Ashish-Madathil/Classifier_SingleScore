import torch.nn as nn
from torchvision.models import resnet34


class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Classifier, self).__init__()
        self.resnet34 = resnet34(pretrained=True)
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet34(x)
