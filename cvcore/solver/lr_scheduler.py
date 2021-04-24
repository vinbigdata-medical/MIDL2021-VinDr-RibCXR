from bisect import bisect_right
import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR



class WarmupMultiStepLR(MultiStepLR):
    """
    Source:

    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py
    """
    def __init__(
        self,
        optimizer,
        milestones,
        iter_per_epoch,
        gamma=0.5,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):

        
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = [m * iter_per_epoch for m in milestones]
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer=optimizer,milestones=milestones,last_epoch= last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCyclicalLR(object):
    """
    Cyclical learning rate scheduler with linear warm-up. E.g.:

    Step mode: ``lr = base_lr * 0.1 ^ {floor(epoch-1 / lr_step)}``.

    Cosine mode: ``lr = base_lr * 0.5 * (1 + cos(iter/maxiter))``.

    Poly mode: ``lr = base_lr * (1 - iter/maxiter) ^ 0.9``.

    Arguments:
        mode (str): one of ('cos', 'poly', 'step').
        base_lr (float): base optimizer's learning rate.
        num_epochs (int): number of epochs.
        iters_per_epoch (int): number of iterations (updates) per epoch.
        warmup_epochs (int): number of epochs to gradually increase learning rate from zero to base_lr.
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0,min_lr=0.):
        self.mode = mode
        assert self.mode in ('cos', 'poly', 'step'), "Unsupported learning rate scheduler"

        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = int(warmup_epochs * iters_per_epoch)
        self.min_lr=min_lr
    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = self.min_lr+0.5 * (self.lr-self.min_lr) * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))

        # warm-up lr scheduler
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
