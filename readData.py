import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
import os
from torch.utils.data import Dataset


class myData(Dataset):
    def __init__(self,  rootDir, imgDir, labelDir):
        self.rootDir = rootDir
        self.imgDir = os.path.join(self.rootDir, imgDir)
        self.imgPath = os.listdir(self.imgDir)
        self.labelDir = os.path.join(self.rootDir, labelDir)

    def __getitem__(self, idx):
        imgName = self.imgPath[idx]
        imgIndex = imgName[:-4]
        imgAbsName = os.path.join(self.imgDir, imgName)
        labelAbsName = os.path.join(self.labelDir, imgName)
        transform = torchvision.transforms.ToTensor()
        img = Image.open(imgAbsName).convert('L')
        img = transform(img)
        img = img
        label = Image.open(labelAbsName).convert('1')
        label = transform(label)
        return img, label

    def __len__(self):
        return len(self.imgPath)

class myData4Test(Dataset):
    def __init__(self,  rootDir, imgDir):
        self.rootDir = rootDir
        self.imgDir = os.path.join(self.rootDir, imgDir)
        self.imgPath = os.listdir(self.imgDir)

    def __getitem__(self, idx):
        imgName = self.imgPath[idx]
        imgIndex = imgName[:-4]
        imgAbsName = os.path.join(self.imgDir, imgName)
        transform = torchvision.transforms.ToTensor()
        img = Image.open(imgAbsName).convert('L')
        # img = img.resize((640, 640))
        img = transform(img)
        return img, imgName

    def __len__(self):
        return len(self.imgPath)

