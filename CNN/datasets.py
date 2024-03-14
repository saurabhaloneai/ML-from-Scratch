import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class WhaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.target_dict = {}

        # Loop through the folders
        for i, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.target_dict[folder] = i  # Assign a target value to each folder
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    self.samples.append((file_path, i))  # (file_path, target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, target = self.samples[idx]
        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image, target

