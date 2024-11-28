

import os
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils_diffusion import denormalize
import torch


def load_data(dataset_name):

    
    if dataset_name == 'MNIST': #std = 0.05
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # Converts to tensor and scales pixel values to [0, 1]
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std of 0.5 for grayscale
        ])
        data_root = './dataset'
        dataset = datasets.MNIST(
            root=data_root,
            train=True,  # Use 'train' or 'test'
            transform=transform,
            download=True   # Downloads the dataset if not available locally
        )

    elif dataset_name == 'CIFAR10':
        
        transform = transforms.Compose([
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        data_root = './dataset'
        dataset = datasets.CIFAR10(
            root=data_root,
            train=True,  # Use 'train' or 'test'
            transform=transform,
            download=True   # Downloads the dataset if not available locally
        )

    elif dataset_name == 'CelebA':
        
        transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 128x128
        transforms.CenterCrop(64),    # Crop to 64x64
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        data_root = './dataset'
        dataset = datasets.CelebA(
            root=data_root,
            split='train',  # Use 'train', 'valid', or 'test'
            transform=transform,
            download=True   # Downloads the dataset if not available locally
        )

    return dataset


def select_dataset(dataset_name, batch_size, device='cuda'):

    dataset = load_data(dataset_name)

    # dataset = data.to(torch.float).to(device)    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                            #  generator=torch.Generator(device=device)
                            )

    
    return dataloader

def get_noise_dataset(total_size, batch_size, device='cuda'):

    niose_file = f'./toy_datasets/niose_{total_size}.pkl'
    p = Path(f'./toy_datasets')
    p.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(niose_file):
        f = open (niose_file, 'wb')
        noise_dataset = np.random.randn(total_size, 2)
        pickle.dump(noise_dataset, f)
        f.close()
    else:
        f = open(niose_file, 'rb')
        noise_dataset = pickle.load(f)
        f.close()  
    noise_dataset = torch.from_numpy(noise_dataset).to(torch.float).to(device)    
    dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


if __name__ == '__main__':
   
    dataset_name = 'MNIST'
    # dataloader = select_dataset('MNIST', 10)
    # dataloader = select_dataset('CIFAR10', 10)
    dataloader = select_dataset('CelebA', 10)
    # Iterate through the dataset
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images Shape: {images.shape}")  # Shape: [batch_size, 3, 128, 128]
        print(f"Labels Shape: {labels.shape}")  # Shape: [batch_size, num_attributes]
        img = denormalize(images)
        # plt.imshow(img[0], cmap='gray')
        plt.imshow(img[0])
        plt.title(labels[0]) 
        plt.axis('off')
        plt.show()
        break
    print()

