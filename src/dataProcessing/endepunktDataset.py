import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

class EndepunktDataset(Dataset):
    def __init__(self, image_dir, dom_dir, transform=None):
        self.image_dir = image_dir
        self.dom_dir = dom_dir
        self.images_file= sorted([f for f in self.image_dir.glob('endepunkt_*.png')])
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor()
        ])

    def_len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_path= self.images_file[idx]
        file_id=image_path.stem
        dom_filename=f"{file_id.replace('endepunkt_','endepunkt_dom')}.png"
        dom_path=self.dom_dir/dom_filename
        img=Image.open(image_path).convert('RGB')
        dom=Image.open(dom_path).convert('L')

        img_tensor=self.transform(img)
        dom_tensor=self.transform(dom)

        combined=torch.cat([img_tensor,dom_tensor],dim=0)
        return combined


