import torch
import torch.nn as nn
import torchvision.models as models


class ExModel(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        # Define 2 CNN architectures ResNet18 and VGG16
        self.resnet18 = models.resnet18(pretrained=True)
        # Freeze the parameters of the model
        for param in self.resnet18.parameters():
            param.requires_grad = False
        # Change the last layer of the model
        self.resnet18.fc = nn.Linear(512, n_classes)

        self.vgg16 = models.vgg16(pretrained=True)
        # Freeze the parameters of the model
        for param in self.vgg16.parameters():
            param.requires_grad = False
        # Change the last layer of the model
        self.vgg16.classifier[6] = nn.Linear(4096, n_classes)

        self.dropout = nn.Dropout(0.5)
        self.combine_layer = nn.Linear(2*n_classes, n_classes)
    


    def forward(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(image)
        
        # Get predictions from VGG16
        vgg_pred = self.vgg16(image)

        resnet_pred = self.dropout(resnet_pred)
        vgg_pred = self.dropout(vgg_pred)
        
        # Concatenate the predictions along the channel dimension
        concat_pred = torch.cat((resnet_pred, vgg_pred), dim=1)
        
        # Combine the predictions using a linear layer
        pred = self.combine_layer(concat_pred)

        return pred
    

