import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torchmetrics import F1Score


from torch.utils.data import DataLoader, Dataset
from models.model import  ResNet18, VGG16
from datasets.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import tqdm

import os

save_model_path = "checkpoints/"
pth_name = "saved_model.pth"
# pth_name = "resnet18sgd.pth"
# pth_name = "resnet18adam.pth"
# pth_name = "vgg16sgd.pth"
# pth_name = "vgg16adam.pth"


def val(model, data_val, loss_function, writer, epoch):
    f1score = 0
    f1 = F1Score(num_classes=107, task = 'multiclass')
    data_iterator = enumerate(data_val)  # take batches
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm.tqdm(total=len(data_val))
        tq.set_description('Validation:')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)

            loss = loss_function(pred, label.float())
            loss = loss.cuda()

            pred = pred.softmax(dim=1)
            
            f1_list.extend(torch.argmax(pred, dim =1).tolist())
            f1t_list.extend(torch.argmax(label, dim =1).tolist())
            #f1score += f1_score(label.squeeze().detach().cpu(), pred.squeeze().detach().cpu())

            total_loss += loss.item()
            tq.update(1)


    writer.add_scalar("Validation mIoU", f1score/len(data_val), epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))


    return None


def train(model, dataloader, val_loader, optimizer, loss_fn, n_epochs):
    device = 'cuda'
    writer = SummaryWriter()

    model.cuda()  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        running_loss = 0.0
        tq = tqdm.tqdm(total=len(dataloader))
        tq.set_description('epoch %d' % (epoch))
        f1score = 0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()
            #f1score += f1_score(labels.detach().cpu(), outputs.detach().cpu())

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        tq.close()
        epoch_loss = running_loss / len(dataloader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
        
        print(f1score/len(dataloader))
        
        val(model, val_loader, loss_fn, writer, epoch)
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)
        model.train()


train_data = custom_dataset("train")
val_data = custom_dataset("val")

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=8
)

# CNN architectures ResNet18 and VGG16 are defined in models/model.py
model = ResNet18(107).cuda()   # Initializing an object of the class.
# model = VGG16(107).cuda()   # Initializing an object of the class.
# print(model(train_data[0][0].unsqueeze(0).cuda()))

# Optimizers are defined in torch.optim
optimizer = SGD(model.parameters(),  lr=0.005)
# optimizer = Adam(model.parameters(), lr=0.0005)
# optimizer = Adam(model.parameters(), lr=0.0001)


# Loss functions are defined in torch.nn.functional
loss = nn.CrossEntropyLoss()

# if you want to load your pretrained model or
# you want to resume stopped training
# use torch.load_state_dict by checking the library!


train(model, train_loader, val_loader,  optimizer,loss, 50)
