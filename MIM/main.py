__author__ = 'Devansh Arpit'
'''
Neural Bayes- Mutual Information Maximization (MIM)

'''
from argparse import Namespace
import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
import pickle as pkl
from models import CNN
from utils import get_loss, get_noise
from data import get_dataset
import tqdm
import sys
import json

parser = argparse.ArgumentParser(description='Neural Bayes-MIM')

# Directories
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')

# Hyperparams
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate ')
parser.add_argument('--bs', type=int, default=4000, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=500, metavar='N',
                    help='minibatch size')
parser.add_argument('--wdecay', type=float, default=0.0000,
                    help='weight decay applied to all weights')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

parser.add_argument('--alpha', type=float, default=4,
                    help='regularization coefficient alpha')
parser.add_argument('--beta', type=float, default=2,
                    help='GP regularization coefficient beta')
parser.add_argument('--all', action='store_true',
                    help='apply MI to all layers')
parser.add_argument('--multiscale', nargs='+', type=int, default=[2],
                    help='apply objective to multiscale. Provide scales as list')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# meta specifications
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name (mnist, cifar10)')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])


args = parser.parse_args()
assert args.bs%args.mbs==0, 'BS should be an integer multiple of MBS'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)
print('Using GPUs: ', args.gpu)

args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, str(os.getpid())) 
print('save_dir ', args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
else:
    print('Folder already exists! Aborting experiment...')
    exit(0)

with open(args.save_dir + '/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
with open(args.save_dir + '/log.txt', 'w') as f:
    f.write(str(os.getpid()) + ' python ' + ' '.join(s for s in sys.argv) + '\n')

# Set random seed
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################
print('==> Preparing data..')
trainloader, _, testloader, _, _ = get_dataset(args)

###############################################################################
# Build the model
###############################################################################
print('==> Building model..')
start_epoch=0
feat_net0 = CNN(bn=True, dataset=args.dataset)
feat_net = torch.nn.DataParallel(feat_net0, device_ids=range(len(args.gpu)))
if use_cuda:
    feat_net.cuda()
params = list(feat_net.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

###############################################################################
# Training/Testing code
###############################################################################
def test(epoch, loader):
    global best_loss, feat_net, args

    feat_net.eval()

    tot_loss = 0
    tot_iters = len(loader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(loader)) 
        if use_cuda:
            inputs, _ = inputs.cuda(), targets.cuda()

        Lx_pre = (feat_net(inputs, args.all))
        loss ,_= get_loss(Lx_pre, args)

        tot_loss += loss.data.cpu().numpy()


    tot_loss = tot_loss/(batch_idx+1)
    # Save checkpoint.
    if tot_loss < best_loss:
        print('Saving best model..')
        state = {
            'feat_net': feat_net0, 
            'epoch': epoch
        }
        with open(args.save_dir + '/best_feat_net.pt', 'wb') as f:
                torch.save(state, f)
        best_loss = tot_loss
    return tot_loss


global_iters=0
def train(epoch):
    global trainloader, global_iters
    global optimizer, feat_net, args, feat_net

    feat_net.train()
    train_loss = 0
    jac_loss = 0


    if not hasattr(train, 'nb_samples_seen'):
        train.nb_samples_seen = 0

    iters=0
    tot_iters = len(trainloader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs0, _ = next(iter(trainloader)) 

        if use_cuda:
            inputs0 = inputs0.cuda()
        noise =  np.random.randn()* get_noise(inputs0)
        inputs = inputs0 + 0.1*noise



        Lx_pre = (feat_net(inputs0, args.all))
        loss, CE  = get_loss(Lx_pre, args)

        tot_loss = loss
        
        tot_loss.backward()
        train_loss += loss.data.cpu().numpy()


        if args.beta>0:
            Lx_pre = (feat_net(inputs.detach()))

            Lx_pre0 = (feat_net(inputs0.detach()))

            noise_ = 0.1*noise.view(noise.size(0), -1)

            Lx0 = Lx_pre0 
            Lx = Lx_pre
            grad_penalty = ( ((Lx-Lx0)**2).sum(dim=1)/(noise_**2).sum(dim=1) ).mean()
            (args.beta*grad_penalty).backward()  
            jac_loss += grad_penalty.data.cpu().numpy()

        if train.nb_samples_seen+args.mbs>=args.bs:
            global_iters+=1
            iters+=1

            # average the gradients from all MBS before parameter update
            for name, variable in feat_net.named_parameters():
                g = variable.grad.data
                g.mul_(1./(1+train.nb_samples_seen/float(args.mbs)))

            train.nb_samples_seen = 0

            optimizer.step()
            optimizer.zero_grad()


        else:
            # accumulate gradients until BS samples are seen
            train.nb_samples_seen += args.mbs

    return train_loss/(batch_idx+1) ,jac_loss/(batch_idx+1)

best_loss= np.inf
def train_fn():
    global epoch, args

    epoch = 0
    while epoch<args.epochs:

        epoch+=1

        loss, jac = train(epoch)
       
        valid_loss = test(epoch, testloader)

        status = str(os.getpid())+' Epoch {}/{} | Loss {:3.4f} | jac {:3.4f}'.format(epoch, args.epochs, loss, jac)
        print (status)

        with open(args.save_dir + '/log.txt', 'a') as f:
            f.write(status + '\n')

        print('-' * 89)

train_fn()

