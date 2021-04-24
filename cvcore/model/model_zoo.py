import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.cuda.amp import autocast
from collections import OrderedDict
import math
import os

import torchvision
import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
import segmentation_models_pytorch as smp
from monai.networks.nets import Unet,BasicUnet 

def build_model(cfg):

    if 'unet()'== cfg.MODEL.NAME:
        model= Unet(dimensions=2,in_channels=cfg.DATA.INP_CHANNEL,out_channels=cfg.MODEL.NUM_CLASSES,\
        channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2),num_res_units=2,dropout=cfg.MODEL.DROPOUT)
    elif 'unet(resnet18)'== cfg.MODEL.NAME:
        model= smp.Unet('resnet18',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet(resnet50)'== cfg.MODEL.NAME:
        model= smp.Unet('resnet50',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet(resnet101)'== cfg.MODEL.NAME:
        model= smp.Unet('resnet101',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet(densenet169)'== cfg.MODEL.NAME:
        model= smp.Unet('densenet169',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet(densenet121)'== cfg.MODEL.NAME:
        model= smp.Unet('densenet121',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',in_channels=cfg.DATA.INP_CHANNEL)
    elif 'deeplabv3(resnet50)' == cfg.MODEL.NAME:
        model=smp.DeepLabV3('resnet50',classes=cfg.MODEL.NUM_CLASSES,in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet(b3)'== cfg.MODEL.NAME:
        model= smp.Unet('efficientnet-b3',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet++()'== cfg.MODEL.NAME:
        model=NestedUNet(cfg)
    elif 'unet++(resnet101)'== cfg.MODEL.NAME:
        model=smp.UnetPlusPlus('resnet101',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet++(b0)' ==cfg.MODEL.NAME:
         model=smp.UnetPlusPlus('efficientnet-b0',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    elif('fpn(b0)'==cfg.MODEL.NAME):
        model=smp.FPN('efficientnet-b0',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet',\
        in_channels=cfg.DATA.INP_CHANNEL,decoder_dropout=cfg.MODEL.DROPOUT )
    elif 'unet(b0)'== cfg.MODEL.NAME:
        model= smp.Unet('efficientnet-b0',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    elif 'unet++(b3)'== cfg.MODEL.NAME:
        model= smp.UnetPlusPlus('efficientnet-b3',classes=cfg.MODEL.NUM_CLASSES,encoder_weights='imagenet'\
        ,in_channels=cfg.DATA.INP_CHANNEL)
    return model