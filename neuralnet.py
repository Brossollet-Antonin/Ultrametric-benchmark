# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:14:53 2019

@author: Antonin
"""

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable 
import pdb

from utils import verbose


# 2 hidden layers MLP with 256 ReLU units in each layers (similar to Chaudhry et al. (2019))
input_size = 100
output_size = 128
hidden_size = 250

class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier.

    Adds methods for "elastic weight consolidation" (EWC) to its subclasses.'''
    def __init__(self):
        super().__init__()

        # -EWC:
        self.ewc_lambda = 0     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True      #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None    #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False     #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0 #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x):
        pass

    #----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, data_batch, allowed_classes=None, collate_fn=None):

        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [data_batch]:       batch of (x,y) to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        data_loader = [(data_batch[0][k],data_batch[1][k]) for k in range(len(data_batch[0]))]

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())

## NN instances, as subclasses of the ContinualLearner class

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


class Bottleneck(ContinualLearner):
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


class ResNet(ContinualLearner):
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




class Net_FCRU(ContinualLearner):
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

class Net_CNN(ContinualLearner):
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

hidden_size = 10

hidden_size = 100


# class Net_FCL(ContinualLearner):
#     def __init__(self, dataset, hidden_size, nonlin):
#         super(Net_FCL, self).__init__()
#         self.dataset = dataset
#         self.input_size = self.dataset.data_sz
#         self.hidden_size = hidden_size[0]

#         self.fc1 = nn.Linear(self.input_size, self.hidden_size)
#         self.fc2 = nn.Linear(self.hidden_size, dataset.num_classes)

#     def forward(self, x):
#         x = x.view(-1, self.input_size)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

class Net_FCL(ContinualLearner):
    def __init__(self, dataset, hidden_sizes, nonlin):
        super(Net_FCL, self).__init__()
        self.dataset = dataset
        self.input_size = self.dataset.data_sz
        self.hs = hidden_sizes
        self.nonlin = nonlin
        self.fc = nn.ModuleList()

        self.fc.append(nn.Linear(self.input_size, self.hs[0]))
        hl_id = 0
        while hl_id < len(self.hs)-1: 
            self.fc.append(nn.Linear(self.hs[hl_id], self.hs[hl_id+1]))
            hl_id += 1
        self.fc.append(nn.Linear(self.hs[hl_id], dataset.num_classes))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for lay_id, lay in enumerate(self.fc):
            x = lay(x)
            if self.nonlin == 'relu':
                x = F.relu(x)
        return x




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# net = Net_CNN('MNIST')
# nbr_para = count_parameters(net)
# print(nbr_para)
