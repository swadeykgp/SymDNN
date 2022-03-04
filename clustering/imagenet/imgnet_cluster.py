import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
import os
# For training
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision
import torchvision.models as models
from torchvision import utils

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

import sys
sys.path.insert(1, '../../core')
from patchutils import *


import faiss 
       
import os
import numpy
import random
import time


torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)
def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  
def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  
def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def data_loader(data_dir, batch_size=1, workers=2, pin_memory=True):
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

net= models.resnet152(pretrained=True)
net.eval()
#for name, layer in net.named_modules():
#    print(name, layer)
def extract_layers(model, layers_to_drop = 1):
    return nn.Sequential(*list(model.children())[:-layers_to_drop])
# lets drop the last 30 layers
def extract_lowernet(net, layers_to_drop=9):
    fms_orignet = extract_layers(model = net, layers_to_drop=layers_to_drop)
    for param in fms_orignet.parameters():
        param.requires_grad = False
    return fms_orignet    
# lets check it out once
fms_orignet = extract_lowernet(net, 9)
for name, layer in fms_orignet.named_modules():
    print(name, layer)

from torchvision import utils


trainset, testset = data_loader('../../../dataset/imagenet', batch_size=1)

#n_clusters=2048
n_clusters=512
patch_size = (2, 2)
#patch_size = (4, 4)
channel_count = 3
repeat = 1 
location=False
stride = 0
t0 = time.time()
kmeans = cluster_patches_inc_imgnet(trainset,fms_orignet, patch_size, n_clusters, channel_count, repeat, stride, location)
dt = time.time() - t0
print('done in %.2fs.' % dt)

query_vectors = np.random.random((10, 4)).astype('float32')
kmeans.index.search(query_vectors.astype(np.float32), 1)[1]
print("****************** Done Image based clustering 512 ************************")
