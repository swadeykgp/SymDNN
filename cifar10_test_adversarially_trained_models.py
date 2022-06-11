from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
#from utils import l2_distance
import math
import torch
import torchvision.transforms as transforms
import numpy as np
import faiss
device = "cpu"

import sys 
sys.path.insert(1, './core')
from patchutils import symdnn_purify, fm_to_symbolic_fm

import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import warnings
warnings.filterwarnings('ignore')

import datetime
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.optim as optim

# https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks
# Define a custom function that will clamp the images between 0 & 1 , without being too harsh as torch.clamp 
def softclamp01(image_tensor):
    image_tensor_shape = image_tensor.shape
    image_tensor = image_tensor.view(image_tensor.size(0), -1)
    image_tensor -= image_tensor.min(1, keepdim=True)[0]
    image_tensor /= image_tensor.max(1, keepdim=True)[0]
    image_tensor = image_tensor.view(image_tensor_shape)
    return image_tensor

def data_to_symbol(perturbed_data, index):
    perturbed_data = softclamp01(perturbed_data)
    pfm = perturbed_data.data.cpu().numpy().copy()
    # Re-classify the perturbed image
    Xsym = symdnn_purify(pfm, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
    return Xsym

# Overloaded load functions
def clean_accuracy_symbolic(index, model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 32):
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size]
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size]
            
            
            output = model(data_to_symbol(x_curr, index))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y_curr[idx]:
                    acc += 1

    return acc / x.shape[0]


# All tests 
def test_acc_ext_ta(atk1, model_name):
    
    print('Model: {}'.format(model_name))
    model = load_model(model_name, norm='Linf')

    print("- Torchattacks", atk1)
    start = datetime.datetime.now()
    adv_images = atk1(images, labels)
    acc = clean_accuracy(model, adv_images, labels, batch_size=32)
    end = datetime.datetime.now()
    print('- Torchattacks Robust Acc non-symbolic: {} ({} ms)'.format(acc, int((end-start).total_seconds()*1000)))
    adv_images = atk1(images, labels)
    start = datetime.datetime.now()
    acc_sym = clean_accuracy_symbolic(index, model, adv_images, labels)
    end = datetime.datetime.now()
    print('- Torchattacks Robust Acc symbolic: {} ({} ms)'.format(acc_sym, int((end-start).total_seconds()*1000)))

if __name__ == '__main__':

    print("torchattacks %s"%(torchattacks.__version__))
    
    # Load standard data
    images, labels = load_cifar10(n_examples=256)
    
    print(images.shape)
    print(labels.shape)
    
    
    channel_count = 3 
    stride = 0
    n_clusters = 2048
    
    patch_size = (2, 2)
    location=False
    
    index = faiss.read_index('./cifar10/kmeans_img_k2_s0_c2048_v1_softclamp.index')
    
    centroid_lut = index.reconstruct_n(0, n_clusters)
    
    
    #model_list_linf = ['Rice2020Overfitting', 'Sehwag2020Hydra', 'Wong2020Fast' , 'Engstrom2019Robustness']
    model_list_linf = [ 'Wong2020Fast' , 'Engstrom2019Robustness']
    
    
    for model_name in model_list_linf:
        model = load_model(model_name, norm='Linf')
        acc = clean_accuracy(model, images, labels)
        print('Model: {}'.format(model_name))
        print('- Standard Acc: {}'.format(acc))
    
        atk1 = torchattacks.CW(model, c=1, lr=0.01, steps=100, kappa=0)
        test_acc_ext_ta(atk1,model_name)
        
        atk1 = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5)
        test_acc_ext_ta(atk1,model_name)
    
        atk1 = torchattacks.FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False)
        test_acc_ext_ta(atk1,model_name)
        
        atk1 = torchattacks.AutoAttack(model, eps=8/255, n_classes=10, version='standard')
        test_acc_ext_ta(atk1,model_name)
        
        
        atk1 = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
        test_acc_ext_ta(atk1,model_name)
        
        atk1 = torchattacks.EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2)
        test_acc_ext_ta(atk1,model_name)
        
        atk1 = torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=100)
        test_acc_ext_ta(atk1,model_name)
        
        atk1 = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=40, scale=10, std=0.1, random_start=True)
        test_acc_ext_ta(atk1,model_name)
