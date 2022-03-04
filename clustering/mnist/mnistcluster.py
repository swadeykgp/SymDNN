from  patchutils_img_noiserobust import cluster_patches_img
import pickle
import torch
import faiss 
import numpy as np

from torchvision import transforms, datasets
import torchvision
import torchvision.transforms as transforms

import os
import numpy
import random

torch.manual_seed(0)
numpy.random.seed(0)
random.seed(0)
apply_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                      transforms.Normalize((0.1309,), (0.2893,))])

trainset = datasets.MNIST(root='../../dataset', train=True, download=True, transform=apply_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
#testset = datasets.MNIST(root='../../dataset', train=False, download=True, transform=transform_test)
num_class = 10
#n_clusters=2048
#n_clusters=512
#n_clusters=256
#n_clusters=128
n_clusters=128
#n_clusters=32
#patch_size = (2, 2)
patch_size = (4, 4)
#patch_size = (8, 8)
channel_count = 1
repeat = 2
location=False
stride = 0
sc = True
kmeans = cluster_patches_img(trainloader, patch_size, n_clusters, channel_count, repeat, stride, sc, location)
