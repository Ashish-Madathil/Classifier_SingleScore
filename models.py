import timm
import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)
        # self.resnet18.fc = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(num_features, num_classes)
        # )

    def forward(self, x):
        return self.resnet18(x)
    
#For our case of 5 classes (0 to 4 scores)
# model = ResNet18Classifier(num_classes=5)
class GoogLeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNetClassifier, self).__init__()
        self.googlenet = models.googlenet(pretrained=True)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        # Load the EfficientNet-B0 model pre-trained on ImageNet
        self.effnet = timm.create_model("efficientnet_b3", pretrained=True)
        
        # Replace the classifier layer to match the number of classes in your dataset
        self.effnet.classifier = nn.Linear(self.effnet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.effnet(x)
    
class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Classifier, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet34(x)

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        # self.resnet50.fc = nn.Linear(num_ftrs, num_classes)        
        # Load pre-trained ResNet50 and remove its classifier
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])  # Remove the classifier
        # Spatially average the output feature map of the modified ResNet50
        self.pool = nn.AdaptiveAvgPool2d((1, 1))          
        # Add dropout and the classifier
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(num_ftrs, num_classes)              


    def forward(self, x):
        # return self.resnet50(x) 
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        
        out = self.classifier(x)
        
        return out       