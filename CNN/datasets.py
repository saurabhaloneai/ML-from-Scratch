import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class WhaleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.target_dict = {}

        # Loop through the folders
        folder_labels = sorted(os.listdir(root_dir))  # Sort the folder names
        for i, folder in enumerate(folder_labels):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.target_dict[folder] = i  # Assign a target value to each folder
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    self.samples.append((file_path, i))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, target = self.samples[idx]
        image = Image.open(file_path).convert('RGB')  # Convert to RGB

        # Resize the image to a fixed size
        resize_transform = transforms.Resize((224, 224))
        image = resize_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, target