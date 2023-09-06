import torch
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms
import pandas as pd
from PIL import Image

class EmbryoDataset(Dataset):
    def __init__ (self, txt_path, transform=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        self.img_list = [line.split()[0] for line in lines]
        self.label_list = [line.split()[1] for line in lines]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__ (self, index):
        img_path = self.img_list[index]
        image = Image.open(img_path).convert('RGB')
        label = self.label_list[index]
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
