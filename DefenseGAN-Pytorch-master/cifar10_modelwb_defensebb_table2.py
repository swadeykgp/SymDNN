#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os

from dataset import Adversarial_Dataset
from util_defense_GAN import adjust_lr, get_z_sets, get_z_star, Resize_Image
from model import CNN
from gan_model import Generator
from torchsummary import summary
import copy

import torchattacks
from torchattacks import *

# In[6]:


import random
import torch.nn.functional as F
import math
import faiss
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import warnings
warnings.filterwarnings('ignore')

# Basic definitions for CIFAR-10 inference

import sys
sys.path.insert(1, '../core')


# In[41]:


batch_size = 1
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Change the dataset folder to the proper location in a new system
testset_cifar = torchvision.datasets.CIFAR10(root='../../dataset', train=False, download=True, transform=transform_test)
testloader_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=batch_size, shuffle=False)
testloader_std_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=4, shuffle=False)

random_indices = list(range(0, len(testset_cifar), 100))
testset_subset = torch.utils.data.Subset(testset_cifar, random_indices)
testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=1, shuffle=False)
testloader_subset_vanilla = torch.utils.data.DataLoader(testset_subset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[10]:


from modeldefs import *

batch_size = 4
in_channel = 3
height = 32
width = 32

display_steps = 20


# ### Load classification model

# In[12]:


# # Send the model to GPU
# model = CNN()



# In[44]:



# In[13]:


# model.load_state_dict(torch.load('./checkpoints/cifar10.pth'))

# model = model.to(device_model)


# ### load defense-GAN model

# In[22]:


learning_rate = 10.0
rec_iters = [200, 500, 1000]
rec_rrs = [10, 15, 20]
decay_rate = 0.1
global_step = 3.0
generator_input_size = 32

INPUT_LATENT = 64 


# In[32]:


device_model = 'cpu'
#device_model = torch.device(0)
# Base model for Cifar-10 (data [0,1]) 
pretrained_base_clampled_gradinit = '../cifar10/cifar10_resnet_gradinit_sc_232.pt'
model = resnet20()
summary(model, input_size = (in_channel,height,width), device = 'cpu')
model.load_state_dict(torch.load(pretrained_base_clampled_gradinit))
model.eval()
model = model.to(device_model)

# ### set parameters

# In[11]:


#device_generator =  'cpu'
device_generator = torch.device(1)

# In[33]:


ModelG = Generator()

generator_path = './defensive_models/gen_cifar10_gp_4999.pth'

ModelG.load_state_dict(torch.load(generator_path))

summary(ModelG, input_size = (INPUT_LATENT,), device = 'cpu')


# In[34]:


ModelG = ModelG.to(device_generator)


# In[35]:


loss = nn.MSELoss()


# ### load test dataset

# In[36]:


# adversarial dataset path
root_dir = './adversarial/'


# In[37]:


# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# ### clean Image

# In[46]:



class BPDAattack(object):
    def __init__(self, model=None, defense=None, device=None, epsilon=None, learning_rate=0.5,
                 max_iterations=100, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.device = device

    def generate(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.

        """

        adv = x.detach().clone()

        lower = np.clip(x.detach().cpu().numpy() - self.epsilon, self.clip_min, self.clip_max)
        upper = np.clip(x.detach().cpu().numpy() + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            #adv_purified = self.defense(adv)
            adv_purified = adv.detach()
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward()

            grad_sign = adv_purified.grad.data.sign()

            # early stop, only for batch_size = 1
            # p = torch.argmax(F.softmax(scores), 1)
            # if y != p:
            #     break

            adv += self.LEARNING_RATE * grad_sign

            adv_img = np.clip(adv.detach().cpu().numpy(), lower, upper)
            adv = torch.Tensor(adv_img).to(self.device)
        return adv


print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

bpda_adversary2 = BPDAattack(model, None, None, epsilon=2/255, learning_rate=0.5, max_iterations=100)
bpda_adversary4 = BPDAattack(model, None, None, epsilon=4/255, learning_rate=0.5, max_iterations=100)
bpda_adversary8 = BPDAattack(model, None, None, epsilon=8/255, learning_rate=0.5, max_iterations=100)
bpda_adversary16 = BPDAattack(model, None, None, epsilon=16/255, learning_rate=0.5, max_iterations=100)
atks = [
    bpda_adversary4.generate,
    bpda_adversary8.generate,
    bpda_adversary16.generate,
    TIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(model, eps=8/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(model, eps=8/255, alpha=2/255, steps=100),
    EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(model, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    CW(model, c=1, lr=0.01, steps=100, kappa=0),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    DeepFool(model, steps=100),
    TIFGSM(model, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(model, eps=4/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(model, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(model, eps=4/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(model, eps=4/255, alpha=2/255, steps=100),
    EOTPGD(model, eps=4/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(model, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(model, eps=4/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(model, eps=4/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    APGD(model, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='ce')
]
atk_id = 0




def test_acc(atk):
    model.eval()
    
    running_corrects = 0
    epoch_size = 0
    
    is_input_size_diff = False
    
    save_test_results = []
    
    for rec_iter in rec_iters:
        for rec_rr in rec_rrs:
            
            for batch_idx, (inputs, labels) in enumerate(testloader_subset_vanilla):
     
                if (atk): 
                    inputs  = atk(inputs, labels) 


                # size change
                if inputs.size(2) != generator_input_size :
    
                    target_shape = (inputs.size(0), inputs.size(1), generator_input_size, generator_input_size)
    
                    data = Resize_Image(target_shape, inputs)
                    data = data.to(device_generator)
    
                    is_input_size_diff = True
    
                else :
                    data = inputs.to(device_generator)
    
                # find z*
    
                _, z_sets = get_z_sets(ModelG, data, learning_rate,                                         loss, device_generator, rec_iter = rec_iter,                                         rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step)
    
                z_star = get_z_star(ModelG, data, z_sets, loss, device_generator)
    
                # generate data
    
                data_hat = ModelG(z_star.to(device_generator)).cpu().detach()
    
                # size back
    
                if is_input_size_diff:
    
                    target_shape = (inputs.size(0), inputs.size(1), height, width)
                    data_hat = Resize_Image(target_shape, data_hat)
    
                # classifier 
                data_hat = data_hat.to(device_model)
    
                labels = labels.to(device_model)
    
                # evaluate 
    
                outputs = model(data_hat)
    
                _, preds = torch.max(outputs, 1)
    
                # statistics
                running_corrects += torch.sum(preds == labels.data)
                epoch_size += inputs.size(0)
    
                if batch_idx % display_steps == 0:
                    print("Attack:", atk)  
                    print('{:>3}/{:>3} average acc {:.4f}\r'                      .format(batch_idx+1, len(testloader_subset_vanilla), running_corrects.double() / epoch_size))
    
                del labels, outputs, preds, data, data_hat,z_star
    
            test_acc = running_corrects.double() / epoch_size
    
            print("Attack:", atk)  
            print('rec_iter : {}, rec_rr : {}, Test Acc: {:.4f}'.format(rec_iter, rec_rr, test_acc))
            
            save_test_results.append(test_acc)

for aattkk in atks:
    test_acc(aattkk)
test_acc(None)
