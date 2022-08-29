# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

from genericpath import exists
import os
from torchvision import transforms, datasets
import torch.utils.data as data
import torch
import numpy as np


def create_loader(batch_size, data_dir, data):
    if data.lower() == 'cifar100':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                      transforms.RandomRotation(15),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        num_classes = 100
        image_size = 32

        return train_loader, test_loader, num_classes, image_size

    if data.lower() == 'cub_200_2011':
        n_class = 200
    elif data.lower() == 'dogs':
        n_class = 120
    elif data.lower() == 'mit67':
        n_class = 67
    elif data.lower() == 'stanford40':
        n_class = 40
    else:
        n_class = 1000

    image_size = 224
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(15),
        transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    val_data_dir = os.path.join(data_dir, 'valid') if os.path.exists(os.path.join(data_dir, 'valid')) else os.path.join(data_dir, 'val')
    testset = datasets.ImageFolder(root=val_data_dir, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                              num_workers=4)
    return train_loader, test_loader, n_class, image_size
