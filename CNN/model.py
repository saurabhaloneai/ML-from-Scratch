#lets build the cnn from scratch by combininig all we learn 

# Create dataset and dataloader
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import WhaleDataset
from sklearn.model_selection import train_test_split

root_dir = '/Users/sasaurabhurabhvaishubhalone/Desktop/ML-from-scratch/CNN/dataset'
dataset = WhaleDataset(root_dir, transform=transforms.ToTensor())  # define your transforms
train_samples, test_samples = train_test_split(dataset.samples, test_size=0.2, random_state=42)

# Create train and test datasets
train_dataset = WhaleDataset(root_dir, transform=transforms.ToTensor())
train_dataset.samples = train_samples

test_dataset = WhaleDataset(root_dir, transform=transforms.ToTensor())
test_dataset.samples = test_samples

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)