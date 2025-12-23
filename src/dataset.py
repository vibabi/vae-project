import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_path, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            attr_path (string): Path to the csv file with attributes.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load attributes CSV
        try:
            self.attr = pd.read_csv(attr_path)
            # CelebA filenames are in the first column "image_id"
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

        # --- SPLIT LOGIC (Standard CelebA partitions) ---
        # Train: 1 - 162770
        # Val:   162771 - 182637
        # Test:  182638 - 202599
        if split == 'train':
            self.attr = self.attr.iloc[:162770]
        elif split == 'val':
            self.attr = self.attr.iloc[162770:182637]
        elif split == 'test':
            self.attr = self.attr.iloc[182637:]
        
        # Reset index so we can access by [0...len]
        self.attr = self.attr.reset_index(drop=True)

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, idx):
        # Get image name from the first column (usually 'image_id')
        img_name = self.attr.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback for systems where path might be slightly different
            print(f"Warning: Image not found {img_path}")
            # Return a black image or handle error
            image = Image.new('RGB', (64, 64))

        if self.transform:
            image = self.transform(image)

        return image