import cv2

import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import ImageDataset as Dataset

from network import RestoreNet

def train_val(cfg):
    # 选取训练设备，cpu或gpu
    if cfg.device == 'cpu':
        device = torch.device('cpu')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device
        device = torch.device('cuda:0')

    # 创建Dataloader
    train_data = Dataset(cfg=cfg, phase='train')
    val_data = Dataset(cfg=cfg, phase='val')

    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # 创建神经网络
    model = RestoreNet().to(device)

    # 优化策略，一般用随机梯度下降SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)

    # 学习率衰减策略
    lf = lambda x: (1 - x / cfg.epochs) * (1.0 - cfg.lr_final) + cfg.lr_final
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 创建损失函数
    loss_function = torch.nn.MSELoss()

    for epoch_i in range(0, cfg.epochs):
        model.train()

        loss_sum = 0
        
        for train_i, (imagex, imagey) in enumerate(train_dataloader):
            imagex = imagex.to(device)
            imagey = imagey.to(device)

            optimizer.zero_grad()

            imagey_pred = model(imagex)

            loss = loss_function(imagey_pred, imagey)
            loss_sum = loss_sum + loss.item()
            #if (train_i+1)%50 == 0:
            #    lr = [x['lr'] for x in optimizer.param_groups]
            #    print('Train Epoch:', epoch_i, 'Step:', train_i, 'Loss:', loss_sum/100, 'Learning Rate:', lr)
            #    loss_sum = 0

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        lr = [x['lr'] for x in optimizer.param_groups]
        print('Train Epoch:', epoch_i, 'Loss:', loss_sum/(train_i+1), 'Learning Rate:', lr)
        scheduler.step()

    model.eval()

    for train_i, (imagex, imagey) in enumerate(val_dataloader):
        imagex = imagex.to(device)
        imagey = imagey #.to(device)

        imagey_pred = model(imagex)

        imagey_pred = np.transpose(imagey_pred.detach().cpu().numpy()[0],(1,2,0))
        imagey = np.transpose(imagey.numpy()[0],(1,2,0))

        cv2.imshow('y', imagey)
        cv2.imshow('y_p', imagey_pred)
        cv2.waitKey()

def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='total number of training epochs')
    parser.add_argument('--device', type=str, default='0', help='e.g. cpu or 0 or 0,1,2,3')
    parser.add_argument('--batch-size', type=int, default=10, help='batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers')
    parser.add_argument('--save-path', type=str, default='weight', help='path to save checkpoint')
    parser.add_argument('--lr-init', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.0001, help='最终学习率')

    parser.add_argument('--image-path', type=str, default='data', help='图像路径')
    parser.add_argument('--crop-size', type=int, default=64, help='图像裁剪尺寸')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_cfg()
    
    train_val(cfg)