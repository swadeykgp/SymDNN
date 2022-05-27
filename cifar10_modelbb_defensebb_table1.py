from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import torchvision.utils
from torchvision import models
import torchattacks
from torchattacks import *
from torchvision import datasets, transforms
import numpy as np
#import matplotlib.pyplot as plt
import random

import faiss
import sys
#sys.path.insert(1, './cifar10')
sys.path.insert(1, './core')
from patchutils import symdnn_purify, fm_to_symbolic_fm

import math
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




np.random.seed(0)
use_cuda=False
device='cpu'
batch_size = 1
batch_size_vanilla = 64 

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#CHANGEME - put the dataset location

testset = torchvision.datasets.CIFAR10(root='../../dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testloader_vanilla = torch.utils.data.DataLoader(testset, batch_size=batch_size_vanilla, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
channel_count = 3 
stride = 0
n_clusters = 2048
patch_size = (2, 2)
location=False

from modeldefs import resnet20
#from modeldefs_states import resnet34
#from modeldefs_holdout import resnet50
#from modeldefs_holdout import WideResNet
#from modeldefs_torchattacks import Holdout, Target

# Base model for Cifar-10 (data [0,1]) 
pretrained_base_clampled_gradinit = './cifar10/cifar10_resnet_gradinit_sc_232.pt'
#pretrained_base_clampled_gradinit = './cifar10_resnet_gradinit_best_day1.pt'
net_std = resnet20()
#net_std = Target() 
#net_std.load_state_dict(torch.load('./adversarial-attacks-pytorch-master/demos/checkpoint/target.pth'))
net_std.load_state_dict(torch.load(pretrained_base_clampled_gradinit))
net_std.eval()

#net_holdout = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
#net_holdout = Holdout()
#net_holdout.load_state_dict(torch.load('./adversarial-attacks-pytorch-master/demos/checkpoint/holdout.pth'))

#net_holdout =  resnet34()
#net_holdout =  resnet34(pretrained=True)
#
net_holdout =  resnet20()
net_holdout.load_state_dict(torch.load('./cifar10/cifar10_holdout.pt'))
net_holdout.eval()
#net_holdout = torch.load('./cifar10_resnet_sym_gradinit_40.pth')

#index = faiss.read_index('./cifar10/kmeans_img_k2_s0_c2048_v0.index')
index = faiss.read_index('./cifar10/kmeans_img_k2_s0_c2048_v1_softclamp.index')
centroid_lut = index.reconstruct_n(0, n_clusters)
# Lets check the kind of prediction the net_std is doing
correct = 0
total = 0
net_std.eval()
# Define a custom function that will clamp the images between 0 & 1 , without being too harsh as torch.clamp 
def softclamp01(image_tensor):
    image_tensor_shape = image_tensor.shape
    image_tensor = image_tensor.view(image_tensor.size(0), -1)
    image_tensor -= image_tensor.min(1, keepdim=True)[0]
    image_tensor /= image_tensor.max(1, keepdim=True)[0]
    image_tensor = image_tensor.view(image_tensor_shape)
    return image_tensor

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

bpda_adversary4 = BPDAattack(net_holdout, None, None, epsilon=4/255, learning_rate=0.5, max_iterations=100)
bpda_adversary8 = BPDAattack(net_holdout, None, None, epsilon=8/255, learning_rate=0.5, max_iterations=100)
bpda_adversary16 = BPDAattack(net_holdout, None, None, epsilon=16/255, learning_rate=0.5, max_iterations=100)
atks = [
    bpda_adversary4.generate,
    bpda_adversary8.generate,
    bpda_adversary16.generate,
    TIFGSM(net_holdout, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(net_holdout, eps=4/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(net_holdout, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(net_holdout, eps=4/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(net_holdout, eps=4/255, alpha=2/255, steps=100),
    EOTPGD(net_holdout, eps=4/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(net_holdout, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(net_holdout, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(net_holdout, eps=4/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(net_holdout, eps=4/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    FAB(net_holdout, eps=4/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    TIFGSM(net_holdout, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(net_holdout, eps=8/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(net_holdout, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(net_holdout, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(net_holdout, eps=8/255, alpha=2/255, steps=100),
    EOTPGD(net_holdout, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(net_holdout, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(net_holdout, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(net_holdout, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(net_holdout, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    CW(net_holdout, c=1, lr=0.01, steps=100, kappa=0),
    FAB(net_holdout, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    Square(net_holdout, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    DeepFool(net_holdout, steps=100),
    TIFGSM(net_holdout, eps=16/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(net_holdout, eps=16/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(net_holdout, eps=16/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(net_holdout, eps=16/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(net_holdout, eps=16/255, alpha=2/255, steps=100),
    EOTPGD(net_holdout, eps=16/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(net_holdout, eps=16/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(net_holdout, eps=16/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(net_holdout, eps=16/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(net_holdout, eps=16/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    FAB(net_holdout, eps=16/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    Square(net_holdout, eps=16/255, n_queries=5000, n_restarts=1, loss='ce')
    ]
atk_id = 0

#CHANGEME - select the number of examples to use - 10 means 1000 images, set 5 for 2000 images
#random_indices = list(range(0, len(testset), 5))
random_indices = list(range(0, len(testset), 20))
print(len(random_indices))
#test_subset = torch.utils.data.Subset(testset, random_indices)
#sub_indices = list(range(5))
testset_subset = torch.utils.data.Subset(testset, random_indices)
testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=batch_size, shuffle=False)
testloader_subset_vanilla = torch.utils.data.DataLoader(testset_subset, batch_size=batch_size_vanilla, shuffle=False)

print("Adversarial Image & Predicted Label for Symbolic inference")
def analyse_internals(atk, atk_id, rlevel1, rlevel2):

    print("-"*70)
    #print(atk)
    atk_name = str(atk).split('(')[0]

    print("Attack params:",atk)
    #exit()
    correct_base_clean = 0
    base_clean = 0 

    correct_base_perturbed = 0
    base_perturbed = 0 

    correct_sym_clean = 0
    sym_clean = 0 

    correct_sym_robust = 0
    sym_robust = 0 
  
    multisym_robust = 0    
    
    randomized_robust = 0    
    
    randomized_fixed_robust = 0    

    total = 0

    net_std.eval()
    for images, labels in testloader_subset_vanilla:
    #for images, labels in testloader_subset:
    #for images, labels in testloader:
        start = time.time()
        #print(" ******++++++++++++++============= Start of Test image ================+++++++++++******")
        #print(" ******++++++++++++++============= Start of Test image ================+++++++++++******")
        # Clean base gradinit inference
        with torch.no_grad():
            X1 = images 
            X1 = softclamp01(X1)  
            y = labels.to(device)
            output = net_std.forward(X1)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    base_clean += 1
                #else:
                    # Whenever there is an error, print the image
                #    print("Misclassification: Model: clean base gradinit. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))

        # Clean symbolic inference
        with torch.no_grad():
            X2 = images 
            X2 = softclamp01(X2)  
            Xsym = symdnn_purify(X2, n_clusters, index, centroid_lut, patch_size, stride, channel_count)           
            output = net_std.forward(Xsym)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    sym_clean += 1
                #else:
                    # Whenever there is an error, print the image
                    #print("Misclassification: Model: clean symbolic gradinit. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))

        # Attacked base gradinit inference 
        X3_in = images 
        X3_in = softclamp01(X3_in)  
        X3 = atk(X3_in, labels)
        output = net_std.forward(X3)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                base_perturbed += 1
            #else:
                # Whenever there is an error, print the image
                #print("Misclassification: Model: perturbed base gradinit model. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))


        # Attacked symbolic inference
        pfm = X3.data.cpu().numpy().copy()
        #print(images.shape) 
        #print(X3.shape) 
        Xsym = symdnn_purify(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count)            
        #print(Xsym.shape) 
        output = net_std.forward(Xsym)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                sym_robust += 1
            #else:
                # Whenever there is an error, print the image
                #print("Misclassification: Model: Perturbed symbolic. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))

#        # randomized purification
#        pfm = X3.data.cpu().numpy().copy()
#        Xsym = symdnn_purify(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count,ana=False, multi=False, instr=False, randomize=True, rlevel=rlevel1, rbalance=True, pdf=None)            
#        output = net_std.forward(Xsym)
#        for idx, i in enumerate(output):
#            if torch.argmax(i) == y[idx]:
#                randomized_robust += 1
#            #else:
#                # Whenever there is an error, print the image
#                #print("Misclassification: Model: Perturbed randomized symbolic. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))



   #     # randomized fixed purification
   #     pfm = X3.data.cpu().numpy().copy()
   #     Xsym = symdnn_purify(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count,ana=False, multi=False, instr=False, randomize=True, rlevel=rlevel2, rbalance=False, pdf=None)            
   #     output = net_std.forward(Xsym)
   #     for idx, i in enumerate(output):
   #         if torch.argmax(i) == y[idx]:
   #             randomized_fixed_robust += 1
   #         #else:

        total += batch_size_vanilla
       
        #print("model BB defense BB Base Gradinit model accuracy:{}".format(100 * float(base_clean) / total))
        #print("model BB defense BB Base Gradinit model symbolic accuracy:{}".format(100 * float(sym_clean) / total))
        #print("model BB defense BB Base Gradinit model accuracy after attack :{}".format(100 * float(base_perturbed) / total))
        #print("model BB defense BB Base Gradinit model symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))
        #print("model BB defense BB  Randomized sym accuracy after attack :{}".format(100 * float(randomized_robust) / total))
        #print("model BB defense BB  Randomized Fixed sym accuracy after attack :{}".format(100 * float(randomized_fixed_robust) / total))
        ##print("model BB defense BB  Randomized Multisym accuracy after attack :{}".format(100 * float(multisym_robust) / total))

        #print(" ******++++++++++++++============= End of Test image:{} ================+++++++++++******".format(total))

    #print('Attack Name: {}'.format(atk))
    #print('Attack prarms: {}'.format(atk))
    print('Defense prarms: {},{}'.format(rlevel1,rlevel2))
    print("model BB defense BB Final Base Gradinit model accuracy:{}".format(100 * float(base_clean) / total))
    print("model BB defense BB Final Base Gradinit model symbolic accuracy:{}".format(100 * float(sym_clean) / total))
    print("model BB defense BB Final Base Gradinit model accuracy after attack :{}".format(100 * float(base_perturbed) / total))
    print("model BB defense BB Final Base Gradinit model symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))
    #print("model BB defense BB Final Randomized sym accuracy after attack :{}".format(100 * float(randomized_robust) / total))
    #print("model BB defense BB Final Randomized Fixed sym accuracy after attack :{}".format(100 * float(randomized_fixed_robust) / total))
    #print("Final Randomized Multisym accuracy after attack :{}".format(100 * float(multisym_robust) / total))

for aattkk in atks:
    #analyse_internals(aattkk, atk_id, 14, 20)
    analyse_internals(aattkk, atk_id, 25, 20)
    atk_id +=1
