__author__ = 'Devansh Arpit'
import os
import sys
import math
import torch.nn as nn
import torch
import numpy as np
import subprocess
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO, BytesIO
import pandas as pd

def get_mul_scale_Lx_pre(Lx_pre, multiscale):
    Lx_pre_ = {}
    i=0
    for key in Lx_pre.keys():
        Lx_pre_[str(i)] = Lx_pre[key]
        i+=1
        for scale in multiscale:
            if len(Lx_pre[key].size())>2:
                Lx_pre_scaled = nn.AvgPool2d( min([scale, Lx_pre[key].size(2)]) )(Lx_pre[key]) 
                Lx_pre_[str(i)] = Lx_pre_scaled
                i+=1
    return Lx_pre_

def get_loss(Lx_pre, args):
    EPSILON=1e-7
    CE=0
    Softmax = nn.Softmax(dim=1)
    
    assert isinstance(Lx_pre, dict), 'Lx should be a dictionary of hidden states'

    if len(args.multiscale)>0:
        Lx_pre = get_mul_scale_Lx_pre(Lx_pre, args.multiscale)
    loss = 0

    N=len(Lx_pre.keys())

    for key in Lx_pre.keys():
        hid = Lx_pre[key]
        Lx = Softmax(hid)
        ELx = torch.mean(Lx, dim=0,keepdim=True) 

        CE_ = - ( ((1./ELx.size(1))*torch.log(ELx) + (1.-1./ELx.size(1))*torch.log(1.-ELx))).sum(1).mean()
        CE += CE_
        loss += -(Lx* torch.log(Lx.detach()+EPSILON) ).sum(dim=1).mean() + (1.+ args.alpha)* CE_
        # detach is not technically required (since gradient is mathematically 0), but saves computation
    return loss/N, CE/N

def get_free_gpu():
    # found this code online at https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560; thanks to the author ptrblck
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    gpu_used = [ int(gpu_df['memory.used'][i].split(' ')[0]) for i in range(len(gpu_df['memory.used']))]
    gpu_used = np.asarray(gpu_used)
    # print(gpu_used)
    gpu_avail = np.where(gpu_used<10)[0]
    # print(gpu_avail)
    return gpu_avail

def get_noise(x):
    sz = x.size()
    x = x.view(x.size(0), -1)
    mn = x.mean(dim=0, keepdim=True)
    x = x-mn
    eps = torch.randint(0,2, (x.size(0), x.size(0))).cuda(). type('torch.cuda.FloatTensor')
    noise = torch.mm(x.t(), eps).t()
    norm = torch.norm(noise, dim=1, keepdim=True)
    assert not np.any(norm.detach().cpu().numpy()==0), '0 norm {}'.format(torch.min(norm))
    noise = noise/norm
    return noise.view(sz)