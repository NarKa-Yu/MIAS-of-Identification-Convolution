import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

def get_cifar10(path="./dataset", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = datasets.CIFAR10(root=path, download=False, transform=transform, train=True)
    dataset_test = datasets.CIFAR10(root=path, download=False, transform=transform, train=False)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataset_train, dataset_test, loader_train, loader_test

def get_cifar100(path="./dataset", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = datasets.CIFAR100(root=path, download=False, transform=transform, train=True)
    dataset_test = datasets.CIFAR100(root=path, download=False, transform=transform, train=False)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataset_train, dataset_test, loader_train, loader_test

def get_tinyimage200(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_root_dir = './dataset/tiny-imagenet-200/train'
    test_root_dir = './dataset/tiny-imagenet-200/test'
    dataset_train = datasets.ImageFolder(root=train_root_dir, transform=transform)
    dataset_test = datasets.ImageFolder(root=test_root_dir, transform=transform)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return dataset_train, dataset_test, loader_train, loader_test


