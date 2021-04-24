import gc
import os
import sys
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score
from cvcore.config import get_cfg_defaults
from cvcore.model import build_model
from cvcore.solver import make_optimizer, WarmupCyclicalLR, WarmupMultiStepLR
from cvcore.utils import setup_determinism, setup_logger, load_checkpoint
from cvcore.tools import parse_args, train_loop, valid_model
from cvcore.data.multi_rib_dataset import make_multi_ribs_dataloader, multi_ribs_dataset
import segmentation_models_pytorch as smp
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
scaler = GradScaler()

def main(args, cfg):
    # Set logger
    logger = setup_logger(
        args.mode,
        cfg.DIRS.LOGS,
        0,
        filename=f"{cfg.NAME}.txt")
    # Declare variables
    best_metric = 0
    start_epoch = 0

    # Define model
    model = build_model(cfg)
    # Define optimizer
    optimizer = make_optimizer(cfg, model)

    # Define loss
    if cfg.LOSS.NAME == "ce":
        train_criterion = nn.CrossEntropyLoss()
    elif cfg.LOSS.NAME=='Bce':
        train_criterion= nn.BCEWithLogitsLoss()
    elif cfg.LOSS.NAME=='dice':
        train_criterion= DiceLoss(sigmoid=True)
    model = model.to(device='cuda')
    model = nn.DataParallel(model)

    # Load checkpoint
    model, start_epoch, best_metric, scheduler = load_checkpoint(args, logger.info, model)

    # Load and split data
    if cfg.DATA.TYPE== 'multilabel':
        if args.mode in ("train", "val"):
            valid_loader= make_multi_ribs_dataloader(cfg,mode='val')
            if args.mode=='train':
                train_loader= make_multi_ribs_dataloader(cfg,mode='train')
        elif args.mode == "test":
            #TODO: Write test dataloader.
            pass
    if args.mode == "train" and scheduler == None:
        if cfg.SOLVER.SCHEDULER == "cyclical":
            scheduler = WarmupCyclicalLR("cos", cfg.SOLVER.BASE_LR, cfg.TRAIN.EPOCHES[-1],
                                        iters_per_epoch=len(train_loader),
                                        warmup_epochs=cfg.SOLVER.WARMUP_LENGTH,min_lr=cfg.SOLVER.MIN_LR)
        elif cfg.SOLVER.SCHEDULER == "step":
            scheduler = WarmupMultiStepLR(
                optimizer=optimizer,
                milestones=cfg.TRAIN.EPOCHES[:-1],
                iter_per_epoch=len(train_loader),
                warmup_factor=cfg.SOLVER.BASE_LR/(cfg.SOLVER.WARMUP_LENGTH * len(train_loader)),
                warmup_iters=cfg.SOLVER.WARMUP_LENGTH * len(train_loader),
                last_epoch=start_epoch if start_epoch else -1)
        else:
            scheduler = None

    if args.mode != "test":
        valid_criterion = train_criterion
        if cfg.METRIC.NAME == 'dice':
            valid_metric = DiceMetric(include_background=True,reduction= 'mean')
    if args.mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES[-1]):
            train_loop(logger.info, cfg, model,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch, scaler)
            _, best_metric = valid_model(logger.info, cfg, model,
                                        valid_loader, valid_criterion,
                                        valid_metric, epoch, best_metric, True)
    elif args.mode == "val":
        valid_model(logger.info, cfg, model,
                    valid_loader, valid_criterion,
                    valid_metric, start_epoch)
    elif args.mode == "test":
        #TODO: Write test function.
        pass


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.opts != "":
        cfg.merge_from_list(args.opts)

    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)