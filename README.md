# Yoga image classifier
Classify the yoga pose being performed in the image. Images link: kaggle.com/datasets/shrutisaxena/yogapose-image-classification-dataset

| Model | Optimizer | Loss | Accuracy F1 | If less classes: Classwise accuracy |
|-----------------------|---|----|------|----|
| ResNet18 | SGD | Loss: 0.0636 | F1 score:  tensor(0.7312) |  |
| ResNet18 | Adam | Loss: 0.1897 | F1 score:  tensor(0.6411)  |  |
| VGG16 | SGD | Loss: 0.0410 | F1 score:  tensor(0.7195) |  |
| VGG16 | Adam |  |  |  |
