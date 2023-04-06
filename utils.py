from torchmetrics.classification import MulticlassF1Score
import torch


def metrics(preds, target):
    metr = MulticlassF1Score(num_classes=3).to('cpu')

    return metr(preds, target)


# Define 2 CNN architectures ResNet18 and VGG16

# class ResNet18(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.resnet18 = models.resnet18(pretrained=True)
#         # Freeze the parameters of the model
#         for param in self.resnet18.parameters():
#             param.requires_grad = False
#         # Change the last layer of the model
#         self.resnet18.fc = nn.Linear(512, n_classes)

#     def forward(self, x):
#         x = self.resnet18(x)
#         return x


# class VGG16(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.vgg16 = models.vgg16(pretrained=True)
#         # Freeze the parameters of the model
#         for param in self.vgg16.parameters():
#             param.requires_grad = False
#         # Change the last layer of the model
#         self.vgg16.classifier[6] = nn.Linear(4096, n_classes)

#     def forward(self, x):
#         x = self.vgg16(x)
#         return x
