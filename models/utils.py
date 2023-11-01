import torch
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms


class DatasetSelector:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.channel_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.squeeze(x)),
            transforms.Lambda(lambda x: torch.stack([x, x, x], 0)),
        ])

        self.intel_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])


    def get_dataset(self):
        if self.dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10("../../data", train=True, download=True, transform=self.transform) 
            test_dataset = datasets.CIFAR10("../../data", train=False, download=True, transform=self.transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
            return combined_dataset
        
        elif self.dataset_name == "mnist":
            train_dataset = datasets.MNIST("../../data", train=True, download=True, transform=self.channel_transform)
            test_dataset = datasets.MNIST("../../data", train=False, download=True, transform=self.channel_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
            return combined_dataset
        
        elif self.dataset_name == "fmnist":
            train_dataset = datasets.FashionMNIST("../../data", train=True, download=True, transform=self.channel_transform)
            test_dataset = datasets.FashionMNIST("../../data", train=False, download=True, transform=self.channel_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
            return combined_dataset
        
        elif self.dataset_name == "intel":
            train_dataset = datasets.ImageFolder(root='../../data/intel/train', transform=self.intel_transform)
            test_dataset = datasets.ImageFolder(root='../../data/intel/test', transform=self.intel_transform)
            combined_dataset = ConcatDataset([train_dataset, test_dataset])
            return combined_dataset

        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
        