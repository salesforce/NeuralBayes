__author__ = 'Devansh Arpit'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn as nn
import torch.nn.init as init


def param_init(module, init='ortho'):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init == 'he':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif init=='ortho':
                nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class CNN(nn.Module):
    def __init__(self, bn=True, dataset='mnist', init='ortho'):
        super(CNN, self).__init__()
        nhiddens = [200,500,700,1000]
        if dataset=='mnist':
            self.channel = 1
            self.sz = 28
        elif 'cifar' in dataset:
            self.channel = 3
            self.sz = 32
        elif dataset=='stl10':
            self.channel = 3
            self.sz = 32 
        self.conv1 = nn.Conv2d(self.channel, nhiddens[0], 3, 1)
        if bn:
            self.bn1 = nn.BatchNorm2d(nhiddens[0])
        else:
            self.bn1 = nn.Sequential()



        self.conv2 = nn.Conv2d(nhiddens[0], nhiddens[1], 3, 1)
        if bn:
            self.bn2 = nn.BatchNorm2d(nhiddens[1])
        else:
            self.bn2 = nn.Sequential()


        self.conv3 = nn.Conv2d(nhiddens[1], nhiddens[2], 3, 1)
        if bn:
            self.bn3 = nn.BatchNorm2d(nhiddens[2])
        else:
            self.bn3 = nn.Sequential()


        self.conv4 = nn.Conv2d(nhiddens[2], nhiddens[3], 3, 1)
        if bn:
            self.bn4 = nn.BatchNorm2d(nhiddens[3])
        else:
            self.bn4 = nn.Sequential()

        param_init(self, init=init)
        self.feat_dim = nhiddens[-1]
        self.nhiddens = nhiddens

    def forward(self, x, ret_hid=False, state=-1):
        # print(x.size())
        hid = {}
        x = x.view(-1, self.channel,self.sz,self.sz)
        x=self.conv1(x)
        
        x = F.relu(self.bn1(x))
        hid['0'] = x
        if state==0:
            return x

        x = F.max_pool2d(x, 2, 2)

        x=self.conv2(x)
        
        x = F.relu(self.bn2(x))
        hid['1'] = x
        if state==1:
            return x


        x=self.conv3(x)
        
        x = F.relu(self.bn3(x))
        hid['2'] = x
        if state==2:
            return x

        x = F.max_pool2d(x, 2, 2)
        # x = nn.AvgPool2d(2,2)(x)
        x=self.conv4(x)

        x = F.relu(self.bn4(x))
        hid['3'] = x
        

        x = nn.AvgPool2d(*[x.size()[2]*2])(x)
        out = x.view(x.size()[0], -1)

        if ret_hid:
            return hid
        return out

class MLPLayer(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, bn=True, act=True, dropout=0.):
        super(MLPLayer, self).__init__()
        self.dropout = dropout
        self.act=act
        if bn:
            fc = nn.Linear(dim_in, dim_out)
            bn_ = nn.BatchNorm1d(dim_out)
            self.layer = [fc, bn_]
        else:
            self.layer = [nn.Linear(dim_in, dim_out)]

        param_init(self, init='ortho')
        self.layer = nn.Sequential(*self.layer)
        
    def forward(self, x):
        if len(x.size())>2:
            x = nn.AvgPool2d(*[x.size()[2]*2])(x)
            x = x.view(x.size()[0], -1)
        x=self.layer(x)
        if self.act:
            x = F.relu((x))
            if self.dropout>0:
                x = nn.Dropout(self.dropout)(x)
        return x
    
class PredNet(nn.Module):
    def __init__(self, dim_inp=None, dim_out=1):
        super(PredNet, self).__init__()
        self.dim_inp = dim_inp
        self.dim_out = dim_out
        layer = nn.Linear(dim_inp, dim_out)
        self.net = nn.Sequential(layer)
        
    def forward(self, x):
        if len(x.size())>2:
            x = nn.AvgPool2d(*[x.size()[2]*2])(x)
            x = x.view(x.size()[0], -1)

        x = x.view(-1, self.dim_inp)
        x = self.net(x)
        return x
  
class MLPNet(nn.Module):
    def __init__(self, dim_inp=None, nhiddens=[500, 500, 500], dim_out=10, bn=True, dropout=0.):
        super(MLPNet, self).__init__()
        self.dim_inp = dim_inp
        self.dropout = dropout
        nhiddens =  nhiddens + [dim_out]
        self.layers = nn.ModuleList([])
        for l in range(len(nhiddens)):
            if l==len(nhiddens)-1:
                if len(nhiddens)==1:
                    layer = MLPLayer(dim_inp, nhiddens[l], False, False)
                else:
                    layer = MLPLayer(nhiddens[l-1], nhiddens[l], False, False)
            elif l==0:
                layer = MLPLayer(dim_inp, nhiddens[l], bn, True)
            else:
                layer = MLPLayer(nhiddens[l-1], nhiddens[l], bn, True)
            self.layers.append((layer))
        
    def forward(self, x):
        if len(x.size())>2:
            x = nn.AvgPool2d(*[x.size()[2]*2])(x)
            x = x.view(x.size()[0], -1)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.Dropout(self.dropout)(x) if self.dropout>0 else x
        x = self.layers[-1](x)
        return x