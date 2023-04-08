import torch
import torch.nn
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
import numpy as np
import random


# dataset = ImageFolder(root="datasets/yoga_dataset", transform=transforms.ToTensor())
# print(len(dataset.classes))

class custom_dataset(Dataset):
    # initialize your dataset class
    def __init__(self, mode="train", image_path="datasets/yoga_dataset", label_path="relative/path/to/your/labels", resize_size=(256, 256)):
        self.mode = mode    # you have to specify which set do you use, train, val or test
        self.image_path = image_path  # you may need this var in getitem
        self.label_path = label_path
        self.resize_size = resize_size

        self.image_list = []
        self.label_list = []

        self.unique_labels = []

        for i in os.listdir(image_path):
            self.unique_labels.append(i)

        for i in os.listdir(image_path):
            for j in os.listdir(self.image_path + "/" + i):
                self.image_list.append(i + "/" + j)
                self.label_list.append(i)

        # # Print image and label list
        # print(self.image_list)
        # print(self.label_list)

        # print the length of the image and label list
        # print("Number of images: ", len(self.image_list))
        # print("Number of labels: ", len(self.label_list))

        # distribute to val train and test
        self.train_images = []
        self.train_labels = []

        self.val_images = []
        self.val_labels = []

        self.test_images = []
        self.test_labels = []

        # Combine the image and label lists into a list of tuples
        data = list(zip(self.image_list, self.label_list))

        # Set the random seed
        random.seed(42)

        # Shuffle the data
        random.shuffle(data)

        # Split the data into train, validation, and test sets
        train_size = int(0.7 * len(data))
        val_size = int(0.1 * len(data))
        test_size = len(data) - train_size - val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]

        # Unpack the train, validation, and test data into separate lists of images and labels
        self.train_images, self.train_labels = zip(*train_data)
        self.val_images, self.val_labels = zip(*val_data)
        self.test_images, self.test_labels = zip(*test_data)

        if (self.mode == "train"):
            self.images = self.train_images
            self.labels = self.train_labels
        elif (self.mode == "val"):
            self.images = self.val_images
            self.labels = self.val_labels
        elif (self.mode == "test"):
            self.images = self.test_images
            self.labels = self.test_labels

    def __getitem__(self, index):
        # getitem is required field for pytorch dataloader. Check the documentation
        image = Image.open(self.image_path + "/" + self.images[index])
        label = self.labels[index]
        # print(np.array(image).shape)
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(degrees=(-30, 30)),
            transforms.Resize(self.resize_size),  # resize the image
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(mean=[0.6187, 0.5749, 0.5616], std=[0.2749, 0.2739, 0.2665])  # normalize the pixel values
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize the pixel values
        ])

        # Mean: tensor([0.6187, 0.5749, 0.5616])
        # STD: tensor([0.2749, 0.2739, 0.2665])
        try:
            image = image.convert('RGB')
            image = transform(image) 
        except Exception as e:    
            image = transform(image)

        label = self.parse_labels(label)
        # label = torch.as_tensor(label)

        # all labels should be converted from any data type to tensor
        # for parallel processing
        return image, label


    def parse_labels(self, label):
        # parse the labels to one hot encoding
        one_hot = [0 for i in range(len(self.unique_labels))]
        one_hot[self.unique_labels.index(label)] = 1

        # convert to tensor
        one_hot_tensor = torch.tensor(one_hot, dtype=float, requires_grad=True)

        return one_hot_tensor


    def __len__(self):
        return len(self.images)


custom_dataset()
