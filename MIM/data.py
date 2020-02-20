__author__ = 'Devansh Arpit'
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as utils

torch.manual_seed(0)

NUM_WORKERS = 0

def get_dataset(args):
    if args.dataset=='cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
        train_sup_set = train_set

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        train_sup_loader = torch.utils.data.DataLoader(train_sup_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)



        testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=NUM_WORKERS)


        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        nb_classes = len(classes)
        dim_inp=32*32
    elif args.dataset=='cifar100':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
        train_sup_set = train_set

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        train_sup_loader = torch.utils.data.DataLoader(train_sup_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)


        testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=NUM_WORKERS)


        nb_classes = 100
        dim_inp=32*32
    elif args.dataset=='stl10':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(root=args.data, split='train+unlabeled', download=True, transform=transform_train)

        train_sup_set = torchvision.datasets.STL10(root=args.data, split='train', download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)
        train_sup_loader = torch.utils.data.DataLoader(train_sup_set, batch_size=args.mbs, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True)
       
        testset = torchvision.datasets.STL10(root=args.data, split='test', download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=NUM_WORKERS)


        nb_classes = 10
        dim_inp=64*64
    return trainloader, train_sup_loader, testloader, nb_classes, dim_inp

