# 🍦 Vanilla PyTorch
import torch
from torch.utils.data import DataLoader

# 👀 Torchvision for CV
from torch.utils.data import Dataset
from torchvision import transforms

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

# 📦 Other Libraries
import glob
from PIL import Image

def to_rgb(image):
    return image.convert("RGB")

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class ImageNet(Dataset):
    def __init__(self, root, split, transform):
        super().__init__()
        self.root = root # root: data/imagenet/
        self.split = split # split: {train, val, test}
        self.transform = transform
        self.list_dirs = glob.glob(f"{root}/{split}/*")
        self.len = len(self.list_dirs)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            pass
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.len
            step = idx.step if idx.step is not None else 1
            images = []
            labels = []
            for i in range(start, stop, step):
                img, label = self.__getitem__(i)
                images.append(img)
                labels.append(label)
            return torch.stack(images), torch.tensor(labels)
        else:
            img = Image.open(self.list_dirs[idx]).convert("RGB")
            img = self.transform(img)
            label = int(self.list_dirs[idx].split("/")[-1].split("_")[-1].split(".")[0])
            return img, label
    
    def __repr__(self):
        return f"ImageNet Dataset: {self.split} split"
    
    def __str__(self):
        return f"ImageNet Dataset: {self.split} split"
    

class ImagenetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data/", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 1000
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
            
    def prepare_data(self):
        # Download the CIFAR-100 dataset
        ImageNet(root=self.data_dir, split="train", transform=self.transform)
        ImageNet(root=self.data_dir, split="val", transform=self.transform)

    def setup(self, stage=None):
        # Load the train & val datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageNet(root=self.data_dir, split="train", transform=self.transform)
            self.val_dataset = ImageNet(root=self.data_dir, split="val", transform=self.transform)
        
        # Load the test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = ImageNet(root=self.data_dir, split="test", transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
# Cifar10
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import AutoAugmentPolicy

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data/", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10
        self.autoaugment_transform = transforms.Compose([
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def prepare_data(self):
        # Download the CIFAR-10 dataset
        CIFAR10(root=self.data_dir, train=True, transform=self.autoaugment_transform, download=True)
        CIFAR10(root=self.data_dir, train=False, transform=self.test_transform, download=True)

    def setup(self, stage=None):
        # Load the train & val datasets
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                CIFAR10(root=self.data_dir, train=True,
                        transform=self.autoaugment_transform, download=False),
                        [45000, 5000], generator=torch.Generator().manual_seed(42)
                )
        
        # Load the test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=self.test_transform, download=False)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data/", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 100
        
        # Transformación estándar para el conjunto de prueba
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # # # AutoAugment para el conjunto de entrenamiento y validación
        # self.autoaugment_transform = transforms.Compose([
        #     transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        
        # Transformación para el conjunto de entrenamiento y validación
        self.train_val_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    def prepare_data(self):
        # Download the CIFAR-100 dataset
        CIFAR100(root=self.data_dir, train=True, transform=self.train_val_transform, download=True)
        CIFAR100(root=self.data_dir, train=False, transform=self.test_transform, download=True)

    def setup(self, stage=None):
        # Load the train & val datasets
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                CIFAR100(root=self.data_dir, train=True, transform=self.train_val_transform, download=False),
                [45000, 5000], generator=torch.Generator().manual_seed(42)
            )
        
        # Load the test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR100(root=self.data_dir, train=False, transform=self.test_transform, download=False)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=2, pin_memory=True)

import matplotlib.pyplot as plt

def plot_images(train_loader, test_loader):
    # Plot 25 images from train dataset
    train_images, _ = next(iter(train_loader))
    train_images = train_images[:25]
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_images[i].permute(1, 2, 0))
        ax.axis('off')
    plt.suptitle('Train Images')
    plt.show()

    # Plot 25 images from test dataset
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:25]
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_images[i].permute(1, 2, 0))
        ax.axis('off')
    plt.suptitle('Test Images')
    plt.show()


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Dataset Loader')
    parser.add_argument('--dataset', type=str,  choices=['cifar100', 'cifar10', 'imagenet'], help='Dataset to be loaded')
    args = parser.parse_args()
    
    # Crear carpeta para almacenar los datos
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    dataset_classes = {
        'cifar100': CIFAR100DataModule,
        'cifar10': CIFAR10DataModule,
        'imagenet': ImagenetDataModule
    }

    if args.dataset not in dataset_classes:
        raise ValueError(f"Invalid dataset name. Available options: {', '.join(dataset_classes.keys())}")
    else:
        if not os.path.exists(f"./data/{args.dataset}/"):
            os.makedirs(f"./data/{args.dataset}/")
        datamodule = dataset_classes[args.dataset](data_dir=f"./data/{args.dataset}/")
        datamodule.prepare_data()
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        print(len(train_loader), len(val_loader), len(test_loader))

        # plot_images(train_loader, test_loader)
    
    
    