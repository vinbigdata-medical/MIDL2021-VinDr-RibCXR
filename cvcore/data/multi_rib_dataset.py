import albumentations
from albumentations import Compose, HorizontalFlip, Normalize, VerticalFlip, Rotate, Resize, ShiftScaleRotate, OneOf, GridDistortion, OpticalDistortion, \
    ElasticTransform, IAAAdditiveGaussianNoise, GaussNoise, MedianBlur,  Blur, CoarseDropout,RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import random
import cv2
import time
class multi_ribs_dataset(Dataset):
    def __init__(self,df,transforms,mode,list_label=[]):
        self.df=df
        self.transforms= transforms
        self.mode=mode
        self.list_label=list_label
    def __len__(self):
        return len(self.df['img'])
    def __getitem__(self,index):
        img_path=self.df['img'][index]
        img=Image.open(img_path)
        img=img.convert('L')
        img=np.asarray(img,dtype=np.float32)
        label0=[]
        for name in self.list_label:
            pts=self.df[name][index]
            label= np.zeros((img.shape[:2]),dtype=np.uint8)
            if pts!='None':
                pts= np.array([[[int(pt['x']),int(pt['y'])]] for pt in pts ])
                label=cv2.fillPoly(label,[pts],255)
            label0.append(label)
        label0=np.stack(label0)
        img=img/255
        label0=label0/255
        label0=label0.transpose((1,2,0))
        img=np.expand_dims(img,axis=2)
        dic=self.transforms[self.mode]\
        ( image=img,mask=label0)
        img=dic['image']
        mask=(dic['mask'].permute(2,0,1))
        return img, mask
def make_multi_ribs_dataloader(cfg,mode='train'):
    df_train= pd.read_json(cfg.DATA.JSON.TRAIN)
    df_val=pd.read_json(cfg.DATA.JSON.VAL)
    list_label=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',\
                         'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
    data_transform= {
        'train': Compose([
            Resize(512,512),
            HorizontalFlip(),
            ShiftScaleRotate(rotate_limit=10),  
            RandomBrightnessContrast(),
            ToTensorV2()
        ]),
        'val': Compose([
            Resize(512,512),
            ToTensorV2()
        ])
    }
    if mode=='train':
        ribs_dataset_train= multi_ribs_dataset(df_train,data_transform,mode,list_label)
        return (DataLoader(dataset=ribs_dataset_train,batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=True))
    elif mode=='val':
        ribs_dataset_val= multi_ribs_dataset(df_val,data_transform,mode,list_label)
        return (DataLoader(dataset=ribs_dataset_val,batch_size=1,shuffle=False))