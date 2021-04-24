import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
import os

from cvcore.utils import AverageMeter, save_checkpoint
from cvcore.solver import WarmupCyclicalLR, WarmupMultiStepLR
from cvcore.model import build_model


def train_loop(_print, cfg, model, train_loader,
               criterion, optimizer, scheduler, epoch, scaler):
    _print(f"\nEpoch {epoch + 1}")
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)
    for i, (image, target) in enumerate(tbar):
        target[target!=0]=1
        target=target.long()
        image = image.to(device='cuda',dtype=torch.float)
        target = target.to(device='cuda')
        with autocast():
            loss = criterion(model(image), target)
            # gradient accumulation
            loss = loss / cfg.SOLVER.GD_STEPS
        scaler.scale(loss).backward()
        # lr scheduler and optim. step
        if (i + 1) % cfg.SOLVER.GD_STEPS == 0:
            # optimizer.step()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            if isinstance(scheduler, WarmupCyclicalLR):
                scheduler(optimizer, i, epoch)
            elif isinstance(scheduler, WarmupMultiStepLR):
                scheduler.step()
        # record loss
        losses.update(loss.item() * cfg.SOLVER.GD_STEPS, target.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (
            losses.avg, optimizer.param_groups[-1]['lr']))

    _print("Train loss: %.5f, learning rate: %.6f" %
           (losses.avg, optimizer.param_groups[-1]['lr']))


