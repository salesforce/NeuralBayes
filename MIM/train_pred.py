
__author__ = 'Devansh Arpit'
'''
Train predictor to evaluate MIM features

'''
from argparse import Namespace
import sys
import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch import autograd
import pickle as pkl
from models import PredNet, CNN, MLPNet
import torch.nn.functional as F
import tqdm
import torch.utils.data as utils
import json
from data import get_dataset

parser = argparse.ArgumentParser(description='Train predictor over MIM features')

# Directories
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to load model from')

# hyper-parameters
parser.add_argument('--state', type=int, default=-1,
                    help='which hidden state to use')
parser.add_argument('--rand', action='store_true',
                    help='use random network instead of loading a stored model for extracting features')
parser.add_argument('--hid', type=int, nargs='+', default=[],
                    help='hid dim if nonlin clf is used')

parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate ')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--wdecay', type=float, default=0.0000,
                    help='weight decay applied to all weights')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# meta specifications
parser.add_argument('--gpu', nargs='+', type=int, default=[0])
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name')
parser.add_argument('--log_tail', type=str, default='',
                    help='string added at the end of log file name')

args = parser.parse_args()
args.mbs=args.bs
args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir) 
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
log_dir = args.save_dir + '/'

log_file = args.save_dir + '/log_pred_'+args.dataset+ args.log_tail +'.txt'
with open(log_file, 'w') as f:
    f.write(str(os.getpid()) +' python ' + ' '.join(s for s in sys.argv) + '\n')

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)
# Set the random seed manually for reproducibility.
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################
print('==> Preparing data..')
_,trainloader, testloader, nb_classes, _ = get_dataset(args)

###############################################################################
# Build the model
###############################################################################
if args.rand:
    print('Using features from randomly initialized model')
    feat_net = CNN(bn=False, dataset=args.dataset)
else:
    print('loading existing model')
    with open(args.save_dir + '/best_feat_net.pt', 'rb') as f:
        best_state = torch.load(f)
        feat_net = best_state['feat_net']

if len(args.hid)==0:
    print('Using logistic regression')
    pred_net = PredNet(dim_inp=feat_net.nhiddens[args.state], dim_out=nb_classes)
else:
    print('Using MLP classifier')
    pred_net = MLPNet(dim_inp=feat_net.nhiddens[args.state], nhiddens = args.hid, dim_out=nb_classes, dropout=args.dropout)

params = list(pred_net.parameters())
pred_net = torch.nn.DataParallel(pred_net, device_ids=range(len(args.gpu)))
feat_net.eval()
if use_cuda:
    feat_net.cuda()
    pred_net.cuda()
criterion = nn.CrossEntropyLoss()


###############################################################################
# Training/Testing code
###############################################################################
def test(loader, save=False, epoch=0):
    global best_acc, args, pred_net, feat_net
    pred_net.eval()
    feat_net.eval()
    correct, total = 0,0
    tot_iters = len(loader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(loader)) 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = pred_net(feat_net(inputs, state=args.state))

            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()


    # Save checkpoint.
    acc = 100.*float(correct)/float(total)

    if save and acc > best_acc:
        best_acc = acc
        print('Saving best model..')
        state = {
            'pred_net': pred_net,
            'epoch': epoch
        }
        with open(args.save_dir + '/best_pred_net.pt', 'wb') as f:
            torch.save(state, f)
    return acc


def train(epoch):
    global trainloader, optimizer, args, feat_net, pred_net
    pred_net.train()
    feat_net.eval()
    correct = 0
    total = 0
    total_loss = 0

    optimizer.zero_grad()
    tot_iters = len(trainloader)
    for batch_idx in tqdm.tqdm(range(tot_iters), total=tot_iters):
        inputs, targets = next(iter(trainloader)) 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 

        outputs = pred_net(feat_net(inputs, state=args.state))

        loss = criterion(outputs, targets)

        total_loss_ = loss 
        total_loss_.backward() 

        total_loss += loss.data.cpu()
        _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        optimizer.step()
        optimizer.zero_grad()


        
    acc = 100.*correct/total
    return total_loss/(batch_idx+1), acc


optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

best_acc =0
epoch = 0
def train_fn():
    global epoch, args, best_acc
    while epoch<args.epochs:
        epoch+=1

        loss, train_acc= train(epoch)
        val_acc = test(testloader, save=True)
        status = str(os.getpid())+' Epoch {}/{} | Loss {:3.4f} | acc {:3.2f} | val-acc {:3.2f} (best {:3.2f})'.\
            format( epoch, args.epochs, loss, train_acc, val_acc, best_acc)
        print (status)

        with open(log_file, 'a') as f:
            f.write(status + '\n')

        print('-' * 89)

train_fn()