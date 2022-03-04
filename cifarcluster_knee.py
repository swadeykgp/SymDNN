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
from kneed import DataGenerator, KneeLocator
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
from modeldefs import *

# Base model for Cifar-10 (data [0,1])
pretrained_base_clampled_gradinit = './cifar10/cifar10_resnet_gradinit_best_day1.pt'
net_std = resnet20()
net_std.load_state_dict(torch.load(pretrained_base_clampled_gradinit))
net_std.eval()


max = 2048
clist = [2,2,4,8,16,20,32,40,64,72,92,128,144,192,256,300,350,430,512,700,800, 900, 1024,1100,1200,1300,1400,1500,1600,1700,1800,2048]
el_list = []
patch_size = (2, 2)
#patch_size = (4, 4)
channel_count = 3
repeat = 1
location=False
stride = 0
index = 1
getdata = True
for n_clusters in clist:
    if index == 1:
        t0 = time.time()
        print("clustering of {} symbols started in {}".format(n_clusters,t0)) 
        kmeans, data = cluster_patches_singleshot(trainloader, patch_size, n_clusters, channel_count, repeat, stride, getdata, location)
        dt = time.time() - t0
        el_list.append(kmeans.obj[-1])
        print("clustering of {} symbols done in {}".format(n_clusters,dt))
    else: 
        t0 = time.time()
        print("clustering of {} symbols started in {}".format(n_clusters,t0)) 
        kmeans = cluster_patches_withdata(data, patch_size, n_clusters, channel_count, repeat, stride, location)
        dt = time.time() - t0
        el_list.append(kmeans.obj[-1])
        print("clustering of {} symbols done in {}".format(n_clusters,dt))
        kneedle1=KneeLocator(clist[:index], el_list, curve='convex', direction='decreasing')
        #number of clusters
        knee_point = kneedle1.knee 
        elbow_point = kneedle1.elbow
        print('Knee: ', knee_point) 
        print('Elbow: ', elbow_point)
        print(kneedle1.x_normalized)
        print(kneedle1.y_normalized)
        print(kneedle1.x_difference)
        print(kneedle1.y_difference)
    index = index + 1
kneedle2=KneeLocator(clist, el_list, curve='convex', direction='decreasing')
#number of clusters
print(kneedle2.x_normalized)
print(kneedle2.y_normalized)
print(kneedle2.x_difference)
print(kneedle2.y_difference)

knee_point = kneedle2.knee 
elbow_point = kneedle2.elbow
print('Knee: ', knee_point) 
print('Knee: ', knee_point) 
print('Elbow: ', elbow_point)
