# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:14:53 2019

@author: Antonin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable 
import pdb


# 2 hidden layers MLP with 256 ReLU units in each layers (similar to Chaudhry et al. (2019))
input_size = 100
output_size = 128
hidden_size = 250

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset):

        super(ResNet, self).__init__()
        self.dataset = dataset
        self.in_planes = 64

        self.conv1 = conv3x3(dataset.n_in_channels, self.in_planes)

        if dataset.n_axes == 1:
            self.bn1 = nn.BatchNorm1d(self.in_planes)
        else:
            self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, dataset.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def resnetN(dataset, type=50):
    if type == 18:
        return ResNet(BasicBlock, [2,2,2,2], dataset)
    elif type == 34:
        return ResNet(BasicBlock, [3,4,6,3], dataset)
    elif type == 50:
        return ResNet(Bottleneck, [3,4,6,3], dataset)
    elif type == 101:
        return ResNet(Bottleneck, [3,4,23,3], dataset)
    else:
        return ResNet(Bottleneck, [3,8,36,3], dataset)




class Net_FCRU(nn.Module):
    def __init__(self):
        super(Net_FCRU, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def layer2(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def layer1(self, x):
        x = x.view(-1, input_size)
        x = self.fc1(x)
        return x

    def layer3(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def layer4(self, x):
        x = x.view(-1, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 2 hidden layers CNN

class Net_CNN(nn.Module):
    def __init__(self, dataset):
        super(Net_CNN, self).__init__()
        self.dataset = dataset
        if dataset.data_origin=='MNIST':
            self.in_axis_dim = 28
        elif dataset.data_origin=='CIFAR10' or dataset.data_origin=='CIFAR100':
            self.in_axis_dim = 32
        elif 'artificial' in dataset.data_origin:
            self.in_axis_dim = dataset.data_sz
        if dataset.n_axes == 2:
            self.conv1 = nn.Conv2d(dataset.n_in_channels,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
        else:
            self.conv1 = nn.Conv1d(dataset.n_in_channels,6,5)
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(6,16,5)

        out_of_conv_axis_dim = ((self.in_axis_dim - 4)//2 - 4)//2
        self.fc1 = nn.Linear(16*out_of_conv_axis_dim**self.dataset.n_axes, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,dataset.num_classes)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        out_of_conv_axis_dim = ((self.in_axis_dim - 4)//2 - 4)//2

        x = x.view(-1, 16*out_of_conv_axis_dim**self.dataset.n_axes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

hidden_size = 100


class Net_FCL(nn.Module):
    def __init__(self, dataset, hidden_size):
        super(Net_FCL, self).__init__()
        self.dataset = dataset
        self.input_size = self.dataset.data_sz
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, dataset.num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# class Net_FCL(nn.Module):
#     def __init__(self, dataset, hidden_sizes):
#         super(Net_FCL, self).__init__()
#         self.dataset = dataset
#         self.input_size = self.dataset.data_sz
#         self.hidden_sizes = hidden_sizes
#         self.fc = []

#         self.fc.append(nn.Linear(self.input_size, hidden_sizes[0]))
#         if len(hidden_sizes) > 1:
#             for hid_id in range(len(hidden_sizes)-1):
#                 self.fc.append(nn.Linear(hidden_sizes[hid_id], hidden_sizes[hid_id+1]))

#         self.fc.append(nn.Linear(hidden_sizes[-1], dataset.num_classes))

#     def forward(self, x):
#         x = x.view(-1, self.input_size)
#         for hidden in self.fc:
#             x = hidden(x)
#         return x



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# net = Net_CNN('MNIST')
# nbr_para = count_parameters(net)
# print(nbr_para)
