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
sys.path.insert(1, './core')


from  patchutils import fm_to_symbolic_fm

import faiss 
       
import os
import numpy
import random
import time

HOWMANY = 2000
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
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def data_loader(data_dir, batch_size=1, workers=2, pin_memory=True):
#    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
#    train_loader = torch.utils.data.DataLoader(
#        train_ds,
#        batch_size=batch_size,
#        shuffle=True,
#        num_workers=workers,
#        pin_memory=pin_memory,
#        sampler=None
#    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    #return train_loader, val_loader
    return val_loader



net= models.resnet152(pretrained=True)
net.eval()
#for name, layer in net.named_modules():
#    print(name, layer)
#CHANGEME
# ********** Change the following - point to the local imagenet folder ********* #
#trainset, testset = data_loader('~/dataset/imagenet', batch_size=1)
#trainset_64, testset_64 = data_loader('~/dataset/imagenet', batch_size=64)
testset = data_loader('~/dataset/imagenet', batch_size=1)
testset_64 = data_loader('~/dataset/imagenet', batch_size=64)
# ********** Change the following - point to the local imagenet folder ********* #

# vanilla model test 
def vanilla_test():
    correct = 0 
    total = 0 
    hm = 0 
    net.eval()
    with torch.no_grad():
        for data in testset_64:
            if hm > HOWMANY:
                break
            hm+=64
            X, y = data
            t0 = time.time() 
            output = net.forward(X)
            dt = time.time() - t0
            #print('done in %.2fs.' % dt) 
            #print(output)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                #else:
                #    # Whenever there is an error, print the image
                #    print("Test Image #: {}".format(total+1))
                #    print("Mispredicted label: {}".format(torch.argmax(i)))
                total += 1
            if total % 128 == 0:  
                print("Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
        print("Final Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
        dt2 = time.time() - t0
        #print('done in %.2fs.' % dt2) 

# vanilla model top-5 test 
def vanilla_top5_test():
    correct = 0 
    total = 0 
    hm = 0 
    net.eval()
    with torch.no_grad():
        for data in testset_64:
            if hm > HOWMANY:
                break
            hm+=64
            X, y = data
            t0 = time.time() 
            output = net.forward(X)
            dt = time.time() - t0
            #print('done in %.2fs.' % dt) 
            #print(output)
            for idx, i in enumerate(output):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    flag = 0
                    if j == y[idx]:
                        correct += 1
                        flag = 1
                        break
                #if flag == 0:
                #    # Whenever there is an error, print the image
                #    print("Test Image #: {}".format(total+1))
                #    print("Mispredicted label: {}".format(torch.argmax(i)))
                total += 1
            if total % 128 == 0:  
                print("Top-5 Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
        print("Top-5 Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
        dt2 = time.time() - t0
        #print('done in %.2fs.' % dt2) 

#index = faiss.read_index("./imagenet/kmeans_img_imgnet_k2_s0_c2048_v0.index")
#n_clusters=2048
#patch_size = (2, 2)
#channel_count = 3
#repeat = 2
#location=False
#stride = 0
#centroid_lut = index.reconstruct_n(0, n_clusters)

# symbolic inference test 
def sym_test():
    correct = 0 
    total = 0 
    net.eval()
    hm = 0 
    with torch.no_grad():
        for data in testset:
            if hm > HOWMANY:
                break
            hm+=1
            X, y = data
            X = X.squeeze() 
            #start = time.process_time()
            Xsym_ = fm_to_symbolic_fm(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            #elapsed = (time.process_time() - start)
            #print('Symbol conversion time:',elapsed)
            Xsym = torch.from_numpy(Xsym_)
            Xsym = Xsym.unsqueeze(0)
            output = net.forward(Xsym.float())
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                #else:
                #    # Whenever there is an error, print the image
                #    print("Test Image #: {}".format(total+1))
                #    print("Mispredicted label: {}".format(torch.argmax(i)))
                total += 1
            if total % 50 == 0:  
                print("Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
    print("Final Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))


# symbolic inference test top-5 
def sym_top5_test():
    correct = 0 
    total = 0 
    net.eval()
    hm = 0 
    with torch.no_grad():
        for data in testset:
            if hm > HOWMANY:
                break
            hm+=1
            X, y = data
            X = X.squeeze() 
            Xsym_ = fm_to_symbolic_fm(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            Xsym = torch.from_numpy(Xsym_)
            Xsym = Xsym.unsqueeze(0)
            output = net.forward(Xsym.float())
            for idx, i in enumerate(output):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    flag = 0
                    if j == y[idx]:
                        correct += 1
                        flag = 1
                        break
                #if flag == 0:
                #    # Whenever there is an error, print the image
                #    print("Test Image #: {}".format(total+1))
                #    print("Mispredicted label: {}".format(torch.argmax(i)))
                total += 1
            if total % 50 == 0:  
                print("Top-5 Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
    print("Final Top-5 Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))


# symbolic inference test top-5 top-3 
def sym_top5_top3sym_test():
    correct = 0 
    total = 0 
    fm1,fm2,fm3 =0,0,0
    net.eval()
    hm = 0 
    with torch.no_grad():
        for data in testset:
            if hm > HOWMANY:
                break
            hm+=1
            X, y = data
            X = X.squeeze() 
            Xsym1_, Xsym2_, Xsym3_  = fm_to_multisym_fm(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count, 3)
            Xsym1 = torch.from_numpy(Xsym1_)
            Xsym1 = Xsym1.unsqueeze(0)
            Xsym2 = torch.from_numpy(Xsym2_)
            Xsym2 = Xsym2.unsqueeze(0)
            Xsym3 = torch.from_numpy(Xsym3_)
            Xsym3 = Xsym3.unsqueeze(0)

            output1 = net.forward(Xsym1.float())
            output2 = net.forward(Xsym2.float())
            output3 = net.forward(Xsym3.float())
            final_pred1, final_pred2, final_pred3 = False, False, False
            for idx, i in enumerate(output1):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    final_pred1 = False
                    if j == y[idx]:
                        final_pred1 = True
                        #print(" multisym FM 1 is correct")
                        fm1 +=1
                        break
            for idx, i in enumerate(output2):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    final_pred2 = False
                    if j == y[idx]:
                        final_pred2 = True
                        #print(" multisym FM 2 is correct")
                        fm2 +=1
                        break
            for idx, i in enumerate(output3):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    final_pred3 = False
                    if j == y[idx]:
                        final_pred3 = True
                        #print(" multisym FM 3 is correct")
                        fm3 +=1
                        break
            
            if final_pred1 == True or final_pred2 == True or final_pred3 == True:
                correct += 1
                #print("Multisym robust",multisym_robust)
            #else:
            #    # Whenever there is an error, print the image
            #    print("Test Image #: {}".format(total+1))
            #    print("Mispredicted label: {}".format(torch.argmax(i)))
            total += 1
        if total % 50 == 0:  
            print("Top-5 Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
    print("Final Top-5 Symbolic Accuracy:{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
# symbolic inference test top-5 top-3 
def sym_top5_msr_test():
    correct = 0
    total = 0
    fm1,fm2,fm3 =0,0,0
    net.eval()
    hm = 0 
    with torch.no_grad():
        for data in testset:
            if hm > HOWMANY:
                break
            hm+=1
            X, y = data
            X = X.squeeze()
            Xsym1_, Xsym2_, Xsym3_  = fm_to_multisym_fm(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count, 3)
            Xsym1 = torch.from_numpy(Xsym1_)
            Xsym1 = Xsym1.unsqueeze(0)
            Xsym2 = torch.from_numpy(Xsym2_)
            Xsym2 = Xsym2.unsqueeze(0)
            Xsym3 = torch.from_numpy(Xsym3_)
            Xsym3 = Xsym3.unsqueeze(0)


            sym_map_id = int(random.randrange(0,3))
            if sym_map_id == 0:
                output = net.forward(Xsym1.float())
            elif sym_map_id == 1:
                output = net.forward(Xsym2.float())
            else:
                output = net.forward(Xsym3.float())

            final_pred = False
            for idx, i in enumerate(output):
                ind = np.argpartition(i, -5)[-5:]
                for j in ind:
                    final_pred = False
                    if j == y[idx]:
                        final_pred = True
                        #print(" multisym FM 1 is correct")
                        break

            if final_pred == True:
                correct += 1
                #print("Multisym robust",multisym_robust)
            #else:
            #    # Whenever there is an error, print the image
            #    print("Test Image #: {}".format(total+1))
            #    print("Mispredicted label: {}".format(torch.argmax(i)))
            total += 1
        if total % 50 == 0:
            print("Top-5 MSR Symbolic Accuracy  :{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))
    print("Final Top-5 MSR Symbolic Accuracy :{} , predicted: {}   ,ground truth: {}".format( round(correct/total, 2), correct,total))

print("Resnet 152, 2048, old patch......")
#vanilla_test()
#vanilla_top5_test()
#sym_test()
#sym_top5_test()
##sym_top5_msr_test()
#
#
#net= models.wide_resnet50_2(pretrained=True)
#net.eval()
#
#
#print("wide_resnet50, 2048, old patch......")
#vanilla_test()
#vanilla_top5_test()
#sym_test()
#sym_top5_test()
##sym_top5_msr_test()
#
#
#net=models.resnext101_32x8d(pretrained=True)
#net.eval()
#
#
#print("resnext101, 2048, old patch......")
#vanilla_test()
#vanilla_top5_test()
#sym_test()
#sym_top5_test()
##sym_top5_msr_test()
#
#net=models.alexnet(pretrained=True)
#net.eval()
#
#print("Alexnet, 2048, old patch......")
#vanilla_test()
#vanilla_top5_test()
#sym_test()
#sym_top5_test()
#
index = faiss.read_index("./imagenet/kmeans_img_mnist_k2_s0_c512_v0.index")
n_clusters=512
patch_size = (2, 2)
channel_count = 3
repeat = 2
location=False
stride = 0
centroid_lut = index.reconstruct_n(0, n_clusters)

# non symbolic is acc is already calculated, skipping


net= models.resnet152(pretrained=True)
net.eval()

print("resnet152, 512, old patch......")
sym_test()
sym_top5_test()
#sym_top5_msr_test()


net= models.wide_resnet50_2(pretrained=True)
net.eval()


print("wide_resnet50, 512, old patch......")
sym_test()
sym_top5_test()
#sym_top5_msr_test()


net=models.resnext101_32x8d(pretrained=True)
net.eval()

print("resnext101, 512, old patch......")
sym_test()
sym_test()
sym_top5_test()
#sym_top5_msr_test()

net=models.alexnet(pretrained=True)
net.eval()

print("alexnet, 512, old patch......")
sym_test()
sym_top5_test()
#sym_top5_msr_test()
