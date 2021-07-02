import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def get_loaders(data_directory, batch_size, image_size, image_crop):
    print('==> Preparing ImageNet dataset..')
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomResizedCrop(image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader

def normalize(image, batch_size):
    mean = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
    std = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
    return (image-mean)/ std