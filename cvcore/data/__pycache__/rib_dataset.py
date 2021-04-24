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
class ribs_dataset(Dataset):
    def __init__(self,df,transforms,fold,mode):
        self.df=df 
        self.transforms= transforms
        self.fold=fold 
        self.mode=mode
    def __len__(self):
        return len(self.df['img'])
    def __getitem__(self,index):
        img_path=self.df['img'][index]
        label_path=self.df['label'][index]
        img=Image.open(img_path)
        img=img.convert('RGB')
        img=np.asarray(img,dtype=np.float32)
        img=img/255
        label=Image.open(label_path)
        label=label.convert('L')
        label=np.asarray(label,dtype=np.float32)
        label=np.expand_dims(label,axis=2)
        label=label/255
        dic=self.transforms[self.mode]( image=img,\
        mask=label)
        return (dic['image']).expand(3,-1,-1), (dic['mask'].permute(2,0,1).squeeze(0))
def make_ribs_dataloader(cfg,mode='train'):
    df= pd.read_csv('data.csv')
    data_transform= {
        'train': Compose([
            #HorizontalFlip(),
            #ShiftScaleRotate(),  

            #Resize(512,512),
            ToTensorV2()
        ]),
        'val': Compose([
            #Resize(256,256),
            ToTensorV2()
        ])
    }
    if mode=='train':
        ribs_dataset_train= ribs_dataset(df,data_transform,fold,mode)
        return (DataLoader(dataset=ribs_dataset_train,batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=True))