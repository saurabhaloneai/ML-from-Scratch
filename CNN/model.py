#lets build the cnn from scratch by combininig all we learn 

# Create dataset and dataloader
from datasets import WhaleDataset

root_dir = '/Users/sasaurabhurabhvaishubhalone/Desktop/ML-from-scratch/CNN/dataset'
dataset = WhaleDataset(root_dir, transform=transform)  # define your transforms
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)