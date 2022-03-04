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
from  patchutils_img_cifar_internals import  fm_to_symbolic_fm, fm_to_symbolic_fm_ana,fm_to_symbolic_fm_ana_robust,fm_to_multisym_fm
import scipy.spatial.distance as dist


np.random.seed(0)
use_cuda=False
device='cpu'
batch_size = 1
batch_size_vanilla = 128 
#  From my training code
random_seed = 1 

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CHANGEME - put the dataset location
testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testloader_vanilla = torch.utils.data.DataLoader(testset, batch_size=batch_size_vanilla, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
channel_count = 3 
stride = 0
n_clusters = 2048
patch_size = (2, 2)
location=False

from modeldefs import *

# Base model for Cifar-10 (data [0,1]) 
pretrained_base_clampled_gradinit = './cifar10/cifar10_resnet_gradinit_sc_232.pt'
net_std = resnet20()
net_std.load_state_dict(torch.load(pretrained_base_clampled_gradinit))
net_std.eval()

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

clevel = [ 0.001, 0.01, 0.1, 1, 10]
atks = [
    CW(net_std, c=0.001, lr=0.01, steps=100, kappa=0),
    CW(net_std, c=0.01, lr=0.01, steps=100, kappa=0),
    CW(net_std, c=0.1, lr=0.01, steps=100, kappa=0),
    CW(net_std, c=1, lr=0.01, steps=100, kappa=0),
    CW(net_std, c=10, lr=0.01, steps=100, kappa=0)
]
atk_id = 0


#CHANGEME - select the number of examples to use - 10 means 1000 images, set 5 for 2000 images
random_indices = list(range(0, len(testset), 5))
print(len(random_indices))
#test_subset = torch.utils.data.Subset(testset, random_indices)
#sub_indices = list(range(5))
testset_subset = torch.utils.data.Subset(testset, random_indices)
testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=1, shuffle=False)
import textdistance

def analyse_internals(atk, atk_id):
    print("Adversarial Image & Predicted Label for Symbolic inference")

    print("-"*70)
    #print(atk)
    atk_name = str(atk).split('(')[0]

    if atk_name == 'CW':
        atk_name +=str(atk_id)
    print(atk_name)
    
 
    print("attack name:",atk_name)
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

    total = 0
    case1, case2, case3, case4, case5, case6 = 0,0,0,0,0,0
    
    cases = []
    multi = []
    #set_diff_list_1 = []
    #edit_dist_list_1 = []
    #l2_dist_list_1 = []

    #set_diff_list_2 = []
    #edit_dist_list_2 = []
    #l2_dist_list_2 = []

    set_diff_list_3 = []
    edit_dist_list_3 = []
    l2_dist_list_3 = []

    #set_diff_list_4 = []
    #edit_dist_list_4 = []
    #l2_dist_list_4 = []

    # all symbols by all symbols   
    #missmap1 = np.zeros((n_clusters,n_clusters), dtype=int)
    #missmap2 = np.zeros((n_clusters,n_clusters), dtype=int)
    #missmap3 = np.zeros((n_clusters,n_clusters), dtype=int)
    #missmap4 = np.zeros((n_clusters,n_clusters), dtype=int)

    net_std.eval()
    #for images, labels in testloader_subset:
    for images, labels in testloader:
        start = time.time()
        print(" ******++++++++++++++============= Start of Test image ================+++++++++++******")
        # Clean base gradinit inference
        with torch.no_grad():
            X1 = images 
            X1 = softclamp01(X1)  
            y = labels.to(device)
            output = net_std.forward(X1)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    base_clean += 1
                    correct_base_clean = 1   
                else:
                    # Whenever there is an error, print the image
                    print("Misclassification: Model: clean base gradinit. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
                    correct_base_clean = 0

        # Clean symbolic inference
        with torch.no_grad():
            X2 = images.squeeze() 
            X2 = softclamp01(X2)  
            Xsym_, symmap_clean = fm_to_symbolic_fm_ana_robust(X2, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            Xsym = torch.from_numpy(Xsym_)
            Xsym = Xsym.unsqueeze(0)
            output = net_std.forward(Xsym.float())
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    sym_clean += 1
                    correct_sym_clean = 1   
                else:
                    # Whenever there is an error, print the image
                    print("Misclassification: Model: clean symbolic gradinit. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
                    correct_sym_clean = 0

        # Attacked base gradinit inference 
        X3_in = softclamp01(images)  
        X3 = atk(X3_in, labels)
        output = net_std.forward(X3)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                base_perturbed += 1
                correct_base_perturbed = 1   
            else:
                # Whenever there is an error, print the image
                print("Misclassification: Model: perturbed base gradinit model. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
                correct_base_perturbed = 0   
        # Attacked symbolic inference
        ppm = X3.squeeze()
        pfm = ppm.data.cpu().numpy().copy()
        Xsym_ , symmap_robust = fm_to_symbolic_fm_ana_robust(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        Xsym = torch.from_numpy(Xsym_)
        Xsym = Xsym.unsqueeze(0)
        output = net_std.forward(Xsym.float())
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                sym_robust += 1
                correct_sym_robust = 1   
            else:
                # Whenever there is an error, print the image
                print("Misclassification: Model: Perturbed symbolic. Test Image #: {}, Mispredicted label: {}".format(total+1, torch.argmax(i)))
                correct_sym_robust = 0

        # randomized multisym challenge
        target = labels.to(device)
        Xsym1_, Xsym2_, Xsym3_  = fm_to_multisym_fm(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count, 3)        
        Xsym1 = torch.from_numpy(Xsym1_)
        Xsym1 = Xsym1.unsqueeze(0) 
        Xsym2 = torch.from_numpy(Xsym2_)
        Xsym2 = Xsym2.unsqueeze(0) 
        Xsym3 = torch.from_numpy(Xsym3_)
        Xsym3 = Xsym3.unsqueeze(0)

        sym_map_id = int(random.randrange(0,3)) 

        if sym_map_id == 0:
            output = net_std.forward(Xsym1.float())   
        elif sym_map_id == 1:
            output = net_std.forward(Xsym2.float()) 
        else:
            output = net_std.forward(Xsym3.float())


        #output1 = net_std.forward(Xsym1.float())
        #output2 = net_std.forward(Xsym2.float())
        #output3 = net_std.forward(Xsym3.float())
        #final_pred1, final_pred2, final_pred3 = False, False, False  
        #for idx, i in enumerate(output1):
        #    if torch.argmax(i) == y[idx]:
        #        final_pred1 = True
        #        multi.append(1)
        #for idx, i in enumerate(output2):
        #    if torch.argmax(i) == y[idx]:
        #        final_pred2 = True
        #        multi.append(2)
        #for idx, i in enumerate(output3):
        #    if torch.argmax(i) == y[idx]:
        #        final_pred3 = True
        #        multi.append(3)
        # 
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                multisym_robust += 1
                #print("Multisym robust",multisym_robust)
            else:
                # Whenever there is an error, print the image
                print("Misclassification: Model: Perturbed randomized multisym symbolic. Test Image #: {}".format(total+1))

        total += 1

        # take minimum of the two for creating the missmaps
        check_len = symmap_clean.size
        if check_len > symmap_robust.size:
            check_len = symmap_robust.size
        print("check_len",check_len)
        
       
        if correct_base_clean:
            if correct_sym_clean:
                if correct_base_perturbed:
                    if correct_sym_robust:
                        case1 += 1
                        print("case 1: all correct - not much info here")
                        cases.append(1)
                        # L2 distance
                        #detach first
                        #tmp_image = perturbed_data.squeeze().detach().cpu().numpy()
 
                        #clean_l2 = (images.squeeze()).reshape(32*32*3)
                        #attacked_l2 = (X3.squeeze()).reshape(32*32*3)
                        #l2_dist = dist.euclidean(clean_l2,attacked_l2)
                        #l2_dist_list_1.append(l2_dist)
                        #print("L2 Dist:",l2_dist)

                        ## set difference distance 
                        #symdiff = list(set(symmap_clean).symmetric_difference(set(symmap_robust)))
                        #set_diff = len(symdiff)
                        #set_diff_list_1.append(set_diff)
                        #print("Sym Diff:",set_diff)
            
                        #sc = ','.join(str(s) for s in set(symmap_clean))
                        #sr = ','.join(str(s) for s in set(symmap_robust))
                        #edit_dist = textdistance.levenshtein.distance(sc, sr)
                        ## edit distance 
                        #edit_dist_list_1.append(edit_dist)
                        #print("edit dist:",edit_dist) 
                    else: # Even base model is performing well against adversarial attack
                        case2 += 1
                        print("case 2: Misprediction by symbolic model after attack - very bad case")
                        cases.append(2)
                        #for i in range(check_len):
                        #    missmap2[symmap_clean[i]][symmap_robust[i]] += 1
                        ## L2 distance
                        ##clean_l2 = (images.squeeze()).reshape(32*32*3)
                        ##attacked_l2 = (X3.squeeze()).reshape(32*32*3)
                        ##l2_dist = dist.euclidean(clean_l2,attacked_l2)
                        ##l2_dist_list_2.append(l2_dist)

                        ## set difference distance 
                        #symdiff = list(set(symmap_clean).symmetric_difference(set(symmap_robust)))
                        #set_diff = len(symdiff)
                        #set_diff_list_2.append(set_diff)
            
                        #sc = ','.join(str(s) for s in set(symmap_clean))
                        #sr = ','.join(str(s) for s in set(symmap_robust))
                        #edit_dist = textdistance.levenshtein.distance(sc, sr)
                        ## edit distance 
                        #edit_dist_list_2.append(edit_dist)
                else:
                    if correct_sym_robust:
                        case3 += 1
                        print("case 3: Our best case: attack damages base model, but ineffective against symbolic model")
                        cases.append(3)
                        #for i in range(check_len):
                        #    missmap3[symmap_clean[i]][symmap_robust[i]] += 1
                        # L2 distance
                        clean_l2 = (images.squeeze()).reshape(32*32*3)
                        attacked_l2 = (X3.squeeze()).reshape(32*32*3)
                        l2_dist = dist.euclidean(clean_l2,attacked_l2)
                        l2_dist_list_3.append(l2_dist)
                        print("L2 Dist:",l2_dist)

                        # set difference distance 
                        symdiff = list(set(symmap_clean).symmetric_difference(set(symmap_robust)))
                        set_diff = len(symdiff)
                        set_diff_list_3.append(set_diff)
                        print("Sym Diff:",set_diff)
            
                        sc = ','.join(str(s) for s in set(symmap_clean))
                        sr = ','.join(str(s) for s in set(symmap_robust))
                        edit_dist = textdistance.levenshtein.distance(sc, sr)
                        # edit distance 
                        edit_dist_list_3.append(edit_dist)
                        print("edit dist:",edit_dist) 
                    else:
                        case4 += 1
                        print("case 4: Misclassification in both symbolic & original after attack - bad case")
                        cases.append(4)
                        #for i in range(check_len):
                        #    missmap4[symmap_clean[i]][symmap_robust[i]] += 1
                        ## L2 distance
                        ##clean_l2 = (images.squeeze()).reshape(32*32*3)
                        ##attacked_l2 = (X3.squeeze()).reshape(32*32*3)
                        ##l2_dist = dist.euclidean(clean_l2,attacked_l2)
                        ##l2_dist_list_4.append(l2_dist)

                        ## set difference distance 
                        #symdiff = list(set(symmap_clean).symmetric_difference(set(symmap_robust)))
                        #set_diff = len(symdiff)
                        #set_diff_list_4.append(set_diff)
            
                        #sc = ','.join(str(s) for s in set(symmap_clean))
                        #sr = ','.join(str(s) for s in set(symmap_robust))
                        #edit_dist = textdistance.levenshtein.distance(sc, sr)
                        ## edit distance 
                        #edit_dist_list_4.append(edit_dist)
                        #print("edit dist:",edit_dist) 
                           
            else:
                case5 += 1
                print("case 5: The symbolic inference failed - note down")
                cases.append(5)
        else:        
            case6 += 1
            print("case 6: All cases where the original standard net_std is incorrect: ignored here ")
            cases.append(6)
               
        print("Distribution of cases: case1:{}, case2:{}, case3:{}, case4:{}, case5:{}, case6:{} of total:{}".format(case1, case2, case3, case4, case5, case6, total))

        print("Base Gradinit model accuracy:{}".format(100 * float(base_clean) / total))
        print("Base Gradinit model symbolic accuracy:{}".format(100 * float(sym_clean) / total))
        print("Base Gradinit model accuracy after attack :{}".format(100 * float(base_perturbed) / total))
        print("Base Gradinit model symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))
        print(" Randomized Multisym accuracy after attack :{}".format(100 * float(multisym_robust) / total))

        print(" ******++++++++++++++============= End of Test image:{} ================+++++++++++******".format(total))

    np.savetxt('cases'+atk_name+'.txt', cases, delimiter =', ', fmt='% 4d')  
    np.savetxt('l2_distances_3'+atk_name+'.txt', l2_dist_list_3, delimiter =', ', fmt='%4.5f')  
    np.savetxt('edit_distances_3'+atk_name+'.txt', edit_dist_list_3, delimiter =', ', fmt='% 4d')  
    np.savetxt('set_difference_3'+atk_name+'.txt', set_diff_list_3, delimiter =', ', fmt='% 4d')  

    print('Attack on different models: {}'.format(atk))
    print("Final Base Gradinit model accuracy:{}".format(100 * float(base_clean) / total))
    print("Final Base Gradinit model symbolic accuracy:{}".format(100 * float(sym_clean) / total))
    print("Final Base Gradinit model accuracy after attack :{}".format(100 * float(base_perturbed) / total))
    print("Final Base Gradinit model symbolic accuracy after attack:{}".format(100 * float(sym_robust) / total))
    print("Final Randomized Multisym accuracy after attack :{}".format(100 * float(multisym_robust) / total))
    print("Final Distribution of cases: case1:{}, case2:{}, case3:{}, case4:{}, case5:{}, case6:{} of total:{}".format(case1, case2, case3, case4, case5, case6, total))

for aattkk in atks:
    analyse_internals(aattkk, atk_id)
    atk_id +=1

