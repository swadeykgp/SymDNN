'''
Purify adversarial images within l_inf <= 16/255
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
import os, imageio
import numpy as np
import argparse
import cv2
from networks import *
from utils import *

parser = argparse.ArgumentParser(description='Purify Images')
parser.add_argument('--dir', default= 'adv_images/')
parser.add_argument('--purifier', type=str, default= 'NRP',  help ='NPR, NRP_resG')
parser.add_argument('--dynamic', action='store_true', help='Dynamic inferrence (in case of whitebox attack)')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.purifier == 'NRP':
    netG = NRP(3,3,64,23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP.pth'))
if args.purifier == 'NRP_resG':
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load('pretrained_purifiers/NRP_resG.pth'))
netG = netG.to(device)
netG.eval()

print('Parameters (Millions):',sum(p.numel() for p in netG.parameters() if p.requires_grad)/1000000)

# Imports for SymDNN paper experiments

import torchattacks
from torchattacks import *
from torchvision import datasets, transforms
import numpy as np
#import matplotlib.pyplot as plt
import random

import faiss
import sys 
sys.path.insert(1, '../../core')
from patchutils import symdnn_purify, fm_to_symbolic_fm

import scipy.spatial.distance as dist

from modeldefs import *

net_std = resnet20()
net_std.load_state_dict(torch.load('./cifar10_resnet_gradinit_sc_232.pt'))
net_std.eval()

np.random.seed(0)
use_cuda=False
device='cpu'
#batch_size = 10
batch_size = 1
batch_size_vanilla = 128 

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#CHANGEME - put the dataset location here

testset = torchvision.datasets.CIFAR10(root='../../../../dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testloader_vanilla = torch.utils.data.DataLoader(testset, batch_size=batch_size_vanilla, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
channel_count = 3 

stride = 0
n_clusters = 2048
patch_size = (2, 2)
location=False

index = faiss.read_index('./kmeans_img_k2_s0_c2048_v1_softclamp.index')
centroid_lut = index.reconstruct_n(0, n_clusters)

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




bpda_adversary2 = BPDAattack(net_std, None, None, epsilon=2/255, learning_rate=0.5, max_iterations=100)
bpda_adversary4 = BPDAattack(net_std, None, None, epsilon=4/255, learning_rate=0.5, max_iterations=100)
bpda_adversary8 = BPDAattack(net_std, None, None, epsilon=8/255, learning_rate=0.5, max_iterations=100)
bpda_adversary16 = BPDAattack(net_std, None, None, epsilon=16/255, learning_rate=0.5, max_iterations=100)
atks = [
    bpda_adversary4.generate,
    bpda_adversary8.generate,
    bpda_adversary16.generate,
    TIFGSM(net_std, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(net_std, eps=8/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(net_std, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(net_std, eps=8/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(net_std, eps=8/255, alpha=2/255, steps=100),
    EOTPGD(net_std, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(net_std, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
    APGD(net_std, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(net_std, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(net_std, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    CW(net_std, c=1, lr=0.01, steps=100, kappa=0),
    FAB(net_std, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
    FAB(net_std, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
    Square(net_std, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
    DeepFool(net_std, steps=100),
    TIFGSM(net_std, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5),
    AutoAttack(net_std, eps=4/255, n_classes=10, version='standard'), # take this at last if time permits
    DIFGSM(net_std, eps=4/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9),
    MIFGSM(net_std, eps=4/255, alpha=2/255, steps=100, decay=0.1),
    RFGSM(net_std, eps=4/255, alpha=2/255, steps=100),
    EOTPGD(net_std, eps=4/255, alpha=2/255, steps=100, eot_iter=2),
    APGD(net_std, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
    APGDT(net_std, eps=4/255, steps=100, eot_iter=1, n_restarts=1),
    Jitter(net_std, eps=4/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True),
    APGD(net_std, eps=4/255, steps=100, eot_iter=1, n_restarts=1, loss='ce')
]
random_indices = list(range(0, len(testset), 10))
print(len(random_indices))
testset_subset = torch.utils.data.Subset(testset, random_indices)
testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=batch_size, shuffle=False)

def analyse_internals(atk):
    print("Adversarial Image & Predicted Label for Symbolic inference")

    print("-"*70)
    #print(atk)
    atk_name = str(atk).split('(')[0]

    print("attack name:",atk_name)
    
    nrp = 0 

    nrp_perturb = 0 
    sym_robust = 0 

    total = 0
    
    net_std.eval()
    for images, labels in testloader_subset:
    #for images, labels in testloader:
        # First check the effect on clean accuracy
        img_clean = softclamp01(images) # We use the same transform for SymDNN  
        y = labels
           
        # Now lets purify
        if args.dynamic:
            eps = 16/255
            img_m_clean = img_clean + torch.randn_like(img_clean) * 0.05
            #  Projection
            img_m_clean = torch.min(torch.max(img_m_clean, img_clean - eps), img_clean + eps)
            img_m_clean = torch.clamp(img_m, 0.0, 1.0)
        else:
            img_m_clean = img_clean

        purified = netG(img_m_clean).detach()

        # Try clean inference

        output = net_std.forward(purified)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                nrp += 1
                #print("correct",nrp) 
            #else:
                # Whenever there is an error, print the image
                #print("Misclassification: NRP clean image. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))

         
        # Attacked  inference 
        X3_in = softclamp01(images) # We use the same transform for SymDNN  
        img = atk(X3_in, labels)

        # Now lets purify
        if args.dynamic:
            eps = 16/255
            img_m = img + torch.randn_like(img) * 0.05
            #  Projection
            img_m = torch.min(torch.max(img_m, img - eps), img + eps)
            img_m = torch.clamp(img_m, 0.0, 1.0)
        else:
            img_m = img

        purified_attacked = netG(img_m).detach()

        ## Try attacked inference

        #output = net_std.forward(purified_attacked)
        #for idx, i in enumerate(output):
        #    if torch.argmax(i) == y[idx]:
        #        nrp_perturb += 1
        #        #print("correct",nrp_perturb) 
        #    #else:
        #        # Whenever there is an error, print the image
        #        #print("Misclassification: NRP attacked image. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
        #
        # Attacked symbolic inference
        ppm = purified_attacked
        pfm = ppm.data.cpu().numpy().copy()
        rlevel1 = 25
        Xsym = symdnn_purify(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count,ana=False, multi=False, instr=False, randomize=True, rlevel=rlevel1, rbalance=True, pdf=None)
        output = net_std.forward(Xsym)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                sym_robust += 1
            #else:
                # Whenever there is an error, print the image
                #print("Misclassification: Model: Perturbed symbolic. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
                #correct_sym_robust = 0

        total = total + batch_size

        print(atk_name+": clean nrp acuracy:{}".format(100 * float(nrp) / total))
        print(atk_name+": nrp accuracy after attack:{}".format(100 * float(nrp_perturb) / total))
        print(atk_name+": nrp + symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))

        print(" ******++++++++++++++============= End of Test image:{} ================+++++++++++******".format(total))

    print('Attack on different models: {}'.format(atk))
    print(atk_name+":Final clean nrp acuracy:{}".format(100 * float(nrp) / total))
    print(atk_name+":Final nrp accuracy after attack:{}".format(100 * float(nrp_perturb) / total))
    print(atk_name+": nrp + symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))

for aattkk in atks:
    analyse_internals(aattkk)

