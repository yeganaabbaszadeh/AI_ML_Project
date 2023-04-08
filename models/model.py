import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        # Define a single CNN architecture ResNet18
        self.resnet18 = models.resnet18(pretrained=True)  # Load a pretrained ResNet18 model
        # Replace the last fully connected layer with a new one
        self.resnet18.fc = nn.Linear(512, 1024)  
        self.relu = nn.ReLU(inplace=True)
        self.newfc = nn.Linear(1024, n_classes) # Change the number of output units to n_classes

    def forward(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(image)
        # Apply ReLU activation inplace
        resnet_pred = self.relu(resnet_pred)
        resnet_pred = self.newfc(resnet_pred)

        return resnet_pred


class VGG16(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        # Define a single CNN architecture VGG16
        self.vgg16 = models.vgg16(pretrained=True)  # Load a pretrained VGG16 model
        # Replace the last fully connected layer with a new one
        self.vgg16.classifier[6] = nn.Linear(4096, 1024)  
        self.relu = nn.ReLU(inplace=True)
        self.newfc = nn.Linear(1024, n_classes) # Change the number of output units to n_classes


    def forward(self, image):
        # Get predictions from VGG16
        vgg_pred = self.vgg16(image)
        # Apply ReLU activation inplace
        vgg_pred = self.relu(vgg_pred)
        vgg_pred = self.newfc(vgg_pred)

        return vgg_pred
    
