from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CatDataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        count = 0
        for path in os.listdir(root_dir):
            if os.path.isfile(os.path.join(root_dir, path)):
                count += 1
        self.img_ids = np.linspace(1, count, count)
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                     
        image = io.imread(str(self.root_dir) + "/" + str(idx) + ".jpg")
        
        return image
        
        