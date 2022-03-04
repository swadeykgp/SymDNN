import sys
sys.path.insert(1, './core')
import pickle
import torch
import faiss
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import numpy
import random
import time

import sys 
sys.path.insert(1, './core')
from patchutils import symdnn_purify, fm_to_symbolic_fm

from modeldefs import *
from patchutils import *

torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)
batch_size = 1
num_class = 10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CHANGEME - put the dataset location 
trainset = torchvision.datasets.CIFAR10(root='../dataset/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Load model for clustering
pretrained_base_clampled_gradinit = './cifar10/cifar10_resnet_gradinit_best_day1.pt'
net_std = resnet20()
net_std.load_state_dict(torch.load(pretrained_base_clampled_gradinit))
net_std.eval()

#n_clusters=256
#n_clusters=512
#n_clusters=1024
#n_clusters=256
#n_clusters=128
patch_size = (2, 2)
#patch_size = (4, 4)
#channel_count = 64
channel_count = 3
repeat = 2
location=False
stride = 0
index = 0
getdata = True
clist = [2,2,4,8,16,20,32,40,64,72,92,128,144,192,256,300,350,430,512,700,800, 900, 1024,1100,1200,1300,1400,1500,1600,1700,1800,2048]
for n_clusters in clist:
    t0 = time.time()
    print("clustering of {} symbols started in {}".format(n_clusters,t0))
    if index == 0:
        kmeans, data = cluster_patches_singleshot(trainloader, patch_size, n_clusters, channel_count, repeat, stride, getdata, location)
    else:
        kmeans = cluster_patches_withdata(data, patch_size, n_clusters, channel_count, repeat, stride, location)
    dt = time.time() - t0
    print("clustering of {} symbols done in {}".format(n_clusters,dt))
