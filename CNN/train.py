

from model import CNN
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

#optimizer and loss function for cnn - > multiclass classification\

import torch.optim as optim
from torch import nn

# Create model, loss function and optimizer

model = CNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}")
    

#---------------------Saving the model--------------------------------#

model_path ='/Users/sasaurabhurabhvaishubhalone/Desktop/ML-from-scratch/CNN/weights.pth'

#save the model

torch.save(model.state_dict(), model_path)


##---------------------------testing-------------------------------------##

#show th prediction using matplotlib

import matplotlib.pyplot as plt
import numpy as np

#load the model

model = CNN(num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Get a batch of test data
inputs, labels = next(iter(test_dataloader))

# Make predictions
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)


##---------------------------plotting-------------------------------------##

# Plot the results
fig, axes = plt.subplots(1, 4, figsize=(10, 5))

for i in range(4):
    axes[i].imshow(np.transpose(inputs[i], (1, 2, 0)))
    axes[i].set_title(f'Predicted: {predicted[i]}, Actual: {labels[i]}')
    axes[i].axis('off')
    plt.show()