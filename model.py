import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)

N_FILTERS = 64  # number of filters used in conv_block
K_SIZE = 3  # size of kernel
MP_SIZE = 2  # size of max pooling
EPS = 1e-8  # epsilon for numerical stability

softplus = torch.nn.Softplus()


class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        self.args = args
        self.meta_learner = Net(args.in_channels, args.num_classes, dataset=args.dataset)

    def forward(self, X, adapted_params=None):
        out = self.meta_learner(X, adapted_params)
        return out

    def cloned_state_dict(self):
        adapted_params = OrderedDict()
        for key, val in self.named_parameters():
            adapted_params[key] = val

        return adapted_params


class Net(nn.Module):
    def __init__(self, in_channels, num_classes, dataset='Omniglot'):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.dataset = dataset
        self.num_classes = num_classes
        self.features = nn.Sequential(
            conv_block(0, in_channels, padding=1, pooling=True),
            conv_block(1, N_FILTERS, padding=1, pooling=True),
            conv_block(2, N_FILTERS, padding=1, pooling=True),
            conv_block(3, N_FILTERS, padding=1, pooling=True))
        self.add_module('fc', nn.Linear(64, num_classes))

    def forward(self, X, params=None):
        if params == None:
            out = self.features(X)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        else:
            out = X
            for i in range(4):
                out = F.conv2d(
                    out,
                    params['meta_learner.features.%d.conv%d.weight'%(i,i)],
                    params['meta_learner.features.%d.conv%d.bias'%(i,i)],
                    padding=1)
                out = F.batch_norm(
                    out,
                    self.state_dict()['features.%d.bn%d.running_mean'%(i,i)],
                    self.state_dict()['features.%d.bn%d.running_var'%(i,i)],
                    params['meta_learner.features.%d.bn%d.weight'%(i,i)],
                    params['meta_learner.features.%d.bn%d.bias'%(i,i)],
                    momentum=1,
                    training=True)
                out = F.relu(out, inplace=True)
                out = F.max_pool2d(out, MP_SIZE)

            out = out.view(out.size(0), -1)
            out = F.linear(out, params['meta_learner.fc.weight'],
                           params['meta_learner.fc.bias'])

        out = F.log_softmax(out, dim=1)
        return out


def conv_block(index,
               in_channels,
               out_channels=N_FILTERS,
               padding=0,
               pooling=True):
    seq_dict = OrderedDict([
                ('conv'+str(index), nn.Conv2d(in_channels, out_channels, K_SIZE, padding=padding)),
                ('bn'+str(index), nn.BatchNorm2d(out_channels, momentum=1, affine=True)),
                ('relu'+str(index), nn.ReLU(inplace=True))
            ])
    if pooling:
        seq_dict['pool'+str(index)] = nn.MaxPool2d(MP_SIZE)
    conv = nn.Sequential(seq_dict)
    return conv
