import os
import shutil
import torch
import numpy as np


def save_checkpoint(state, is_best, root, filename):
    """
    Saves checkpoint and best checkpoint (optionally)
    """
    torch.save(state, os.path.join(root, filename))
    # if is_best:
    #     shutil.copyfile(
    #         os.path.join(
    #             root, filename), os.path.join(
    #             root, 'best_' + filename))

def load_checkpoint(args, log, model):
    if args.load != "":
        if os.path.isfile(args.load):
            log(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            try:
                scheduler = ckpt['scheduler']
            except:
                scheduler = None
            if args.reset:
                start_epoch = 0
                scheduler = None
            if args.clear:
                best_metric = np.inf
            log(
                f"=> loaded checkpoint '{args.load}' \
                    (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            log(f"=> no checkpoint found at '{args.load}'")

    else:
        start_epoch = 0
        best_metric = 0
        scheduler = None

    return model, start_epoch, best_metric, scheduler