#multiclass image classification model and a training script for a custom dataset that pulls from a csv file consisting of image paths in the first column and an integer of any value between 0 and 4 in the second column. Using PyTorch.
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 128x8x8
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, 5) #5 classes
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
