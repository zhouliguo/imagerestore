import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, cfg, phase):
        super().__init__()
        self.image_path_x = glob.glob(os.path.join(cfg.image_path, 'original', '*.png'))
        self.image_path_y = glob.glob(os.path.join(cfg.image_path, 'degradation', '*.png'))

    def __len__(self):
        return len(self.image_path_x)
    
    def __getitem__(self, index):
        image_x = cv2.imread(self.image_path_x[index])
        image_y = cv2.imread(self.image_path_y[index])

        image_x = np.transpose(image_x,(2,0,1)).astype(np.float64)
        image_y = np.transpose(image_y,(2,0,1)).astype(np.float64)

        return image_x, image_y


if __name__ == '__main__':

    dataset = ImageDataset()