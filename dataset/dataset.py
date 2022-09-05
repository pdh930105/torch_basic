# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

from genericpath import exists
import os
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.utils.data import DataLoader
import torch
import numpy as np


def create_loader(config):
    if config.dataset.lower() == 'cifar100':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                      transforms.RandomRotation(15),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_test)

        if config.dist:
            train_sampler =ElasticDistributedSampler(dataset=trainset)
            test_sampler =ElasticDistributedSampler(dataset=testset)
            train_loader = DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler,num_workers=config.workers)
            test_loader = DataLoader(testset, batch_size=config.test_batch_size, sampler=test_sampler,num_workers=config.workers)
        
        else:
            train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
            test_loader = DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.workers)
        
        num_classes = 100
        image_size = 32

        return train_loader, test_loader, num_classes, image_size

    if config.dataset.lower() == 'cifar10':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                      transforms.RandomRotation(15),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_test)

        if config.dist:
            train_sampler =ElasticDistributedSampler(dataset=trainset)
            test_sampler =ElasticDistributedSampler(dataset=testset)
            train_loader = DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler,num_workers=4)
            test_loader = DataLoader(testset, batch_size=config.test_batch_size, sampler=test_sampler,num_workers=4)
        
        else:
            train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
            test_loader = DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.workers)
        
        num_classes = 10
        image_size = 32

        return train_loader, test_loader, num_classes, image_size


    if config.dataset.lower() == 'cub_200_2011':
        n_class = 200
    elif config.dataset.lower() == 'dogs':
        n_class = 120
    elif config.dataset.lower() == 'mit67':
        n_class = 67
    elif config.dataset.lower() == 'stanford40':
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

    trainset = datasets.ImageFolder(root=os.path.join(config.data_path, 'train'), transform=transform_train)
    val_data_path = os.path.join(config.data_path, 'valid') if os.path.exists(os.path.join(config.data_path, 'valid')) else os.path.join(config.data_path, 'val')
    testset = datasets.ImageFolder(root=val_data_path, transform=transform_test)

    if config.dist:
        train_sampler = ElasticDistributedSampler(dataset=trainset)
        test_sampler = ElasticDistributedSampler(dataset=testset)
        train_loader = DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=config.pin_memory, num_workers=config.workers)
        test_loader = DataLoader(testset, batch_size=config.test_batch_size, sampler=test_sampler, pin_memory=config.pin_memory,num_workers=config.workers)
        
    else:
        train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=4)
        

    return train_loader, test_loader, n_class, image_size
