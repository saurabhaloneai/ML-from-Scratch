#lets build the cnn from scratch by combininig all we learn 

# Create dataset and dataloader
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datasets import WhaleDataset
from sklearn.model_selection import train_test_split


#---------------------DATASET AND DATALOADER---------------------#


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



#---------------------CNN MODEL--------------------------------#

from torch import nn

class CNN(nn.Module):
    
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x