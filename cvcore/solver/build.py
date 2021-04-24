import torch


def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay,'initial_lr':lr}]

    if cfg.SOLVER.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(params, lr, eps=cfg.SOLVER.ADAM_EPS)
    elif cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)
    return optimizer