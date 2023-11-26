import torch
import cv2, glob
import random

from torchvision import transforms
from torch.utils.data import  Dataset
from glob import glob

IM_SIZE = 256

class PhotoDatasetAugmented(Dataset):
    def __init__(self, drawings_data_dir, photo_data_dir, train=True):
        self.train=train
        self.drawings_path = glob(drawings_data_dir +'/*.jpeg')
        self.photos_path = [ photo_data_dir + elem.split('/')[-1].split('_')[-2] + '_photo.jpeg' for elem in self.drawings_path]
        self.transforms_initial = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256], antialias=True),
        ])
        self.transforms_fliprotate = transforms.Compose([
            transforms.RandomRotation(180, fill=1.0),
            transforms.RandomHorizontalFlip(0.25),
        ])
        self.transforms_color = transforms.ColorJitter(hue=0.5)
    def __len__(self): return len(self.photos_path)
    def __getitem__(self, ix):
        photo = cv2.cvtColor(cv2.imread(self.photos_path[ix]), cv2.COLOR_BGR2RGB)
        drawing = cv2.cvtColor(cv2.imread(self.drawings_path[ix]), cv2.COLOR_BGR2RGB)
        photo = self.transforms_initial(photo)
        drawing = self.transforms_initial(drawing)
        if self.train:
          if random.random()>0.5:
            photo = self.transforms_color(photo)
          pair_tensor = torch.stack((photo, drawing))
          pair_tensor = self.transforms_fliprotate(pair_tensor)
          photo, drawing = torch.unbind(pair_tensor)
        return photo, drawing
