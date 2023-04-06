import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support

def calculate_mean_std(dataset):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('==> Computing mean and std..')
    for images, _ in dataloader:
        for i in range(3):
            mean[i] += images[:,i,:,:].mean()
            std[i] += images[:,i,:,:].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    print(f'Mean: {mean}')
    print(f'STD: {std}')
    # Mean: tensor([0.6187, 0.5749, 0.5616])
    # STD: tensor([0.2749, 0.2739, 0.2665])

if __name__ == '__main__':
    freeze_support()

    data_path = 'datasets/yoga_dataset'
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    dataset = ImageFolder(data_path, transform=transform)

    calculate_mean_std(dataset)

