import os
import argparse
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, cfg, phase = 'train'):
        super().__init__()
        self.image_path_x = glob.glob(os.path.join(cfg.image_path, 'degrad', '*.jpg'))
        self.image_path_y = glob.glob(os.path.join(cfg.image_path, 'original', '*.jpg'))
        self.transform = transforms.Compose([transforms.RandomCrop(size=(cfg.crop_size, cfg.crop_size))])

        self.phase = phase

    def __len__(self):
        return len(self.image_path_x)
    
    def __getitem__(self, index):
        image_x = cv2.imread(self.image_path_x[index])
        image_y = cv2.imread(self.image_path_y[index])

        image_p = np.concatenate((image_x, image_y), 2)
        image_p = np.transpose(image_p,(2,0,1))
        
        image_p = torch.from_numpy(image_p)/255.0

        if self.phase == 'train':    
            image_p = self.transform(image_p)

        image_x = image_p[:3]
        image_y = image_p[3:]

        return image_x, image_y

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weight', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='最终学习率')

    parser.add_argument('--image-path', type=str, default='data', help='图像路径')
    parser.add_argument('--crop-size', type=int, default=64, help='图像裁剪尺寸')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()

    dataset = ImageDataset(cfg)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    for i, (imagex, imagey) in enumerate(dataloader):
        imagex