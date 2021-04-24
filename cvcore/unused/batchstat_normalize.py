import torch

def batchstat_norm(x):
    '''
    Normalize tensor x of size (N,3,H,W) with batch mean and std.
    '''
    mean = []
    std = []
    for i in range(x.shape[1]):
        xi = x[:,i,:,:]
        mean.append(xi.mean().repeat(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1))
        std.append(xi.std().repeat(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1))
    mean = torch.cat(mean, 1)
    std = torch.cat(std, 1)
    x = (x - mean) / std

    return x