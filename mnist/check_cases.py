import torch
import random
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
#import matplotlib.pyplot as plt
import numpy as np
#import huffman
import math
import faiss

import sys
sys.path.insert(1, '../core')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
import warnings
warnings.filterwarnings('ignore')
from patchutils import *
from torchvision import utils

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom function that will clamp the images between 0 & 1 , without being too harsh as torch.clamp 
def softclamp01(image_tensor):
    image_tensor_shape = image_tensor.shape
    image_tensor = image_tensor.view(image_tensor.size(0), -1)
    image_tensor -= image_tensor.min(1, keepdim=True)[0]
    image_tensor /= image_tensor.max(1, keepdim=True)[0]
    image_tensor = image_tensor.view(image_tensor_shape)
    return image_tensor

def abstract_input(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count):
    Xsym_ = fm_to_symbolic_fm(X.squeeze(), n_clusters, index, centroid_lut, patch_size, stride, channel_count)
    Xsym = torch.from_numpy(Xsym_)
    Xsym = Xsym.unsqueeze(0)
    return Xsym.float()

def abstract_input_ana(X, n_clusters, index, centroid_lut, patch_size, stride, channel_count):
    ana=True 
    multi=False
    instr=False
    pdf=None
    Xsym_, sym_map = fm_to_symbolic_fm(X.squeeze(), n_clusters, index, centroid_lut, patch_size, stride, channel_count ,ana=True, multi=False, instr=False, pdf=None)
    Xsym = torch.from_numpy(Xsym_)
    Xsym = Xsym.unsqueeze(0)
    return Xsym.float(), sym_map

# All we need for symbols
index = faiss.read_index("../clustering/mnist/kmeans_mnist_fullnet_k2_s0_c768_sc.index")
n_clusters=768
patch_size = (2, 2)
channel_count = 1
location=False
patch_stride = 0
centroid_lut = index.reconstruct_n(0, n_clusters)



# All we need for Filter symbols
#index_f = faiss.read_index("../clustering/mnist/kmeans_flt_mnist_k2_s0_c4_sc.index")
index_f = faiss.read_index("../clustering/mnist/kmeans_flt_mnist_k2_s0_c8_sc.index")
#index_f = faiss.read_index("../clustering/mnist/kmeans_flt_mnist_k2_s0_c12_sc.index")

#index_f = faiss.read_index("../clustering/mnist/kmeans_flt_mnist_k2_s0_c16_sc.index")
#index_f = faiss.read_index("../clustering/mnist/kmeans_flt_mnist_k2_s0_c128_sc.index")
n_clusters_f=8
centroid_lut_f = index_f.reconstruct_n(0, n_clusters_f)


def abstract_filter(layer_filter, n_clusters, index, centroid_lut, patch_size, stride, channel_count):
    ana=False 
    multi=False
    instr=False
    pdf=None

    # The shape is n, c, w, h 
    # We need to do some manipulation 
    n, c, w, h = layer_filter.shape
    #print(n,c,w,h)
    buffer_sym = []
    buffer_flt_ch = []
    buffer_sym_ch = []
    for num_filters in range(n):
        flt_ch = layer_filter[num_filters,:,:,:]
        flt_ch = flt_ch.squeeze()
        for num_channels in range(c):
            if c > 1:
                flt = flt_ch[num_channels,:,:]
            else:
                flt = flt_ch
            flt = flt.squeeze()
            #print(flt.shape)
            flt_img, flt_sym = fm_to_symbolic_fm(flt, n_clusters, index, centroid_lut, patch_size, stride, channel_count,ana=True, multi=False, instr=False, pdf=None)
            #print(len(flt_sym))
            buffer_flt_ch.append(flt_img)
            buffer_sym_ch.append(flt_sym)
        buffer_sym.append(buffer_sym_ch)
        buffer_sym_ch =[]
    abstract_flt =  np.array(buffer_flt_ch, dtype=np.double)
    #symbolic_flt =  np.array(buffer_sym_ch, dtype=np.int)
    abstract_flt_t =  torch.from_numpy(abstract_flt)
    #symbolic_flt_t =  torch.from_numpy(symbolic_flt)
    abstract_flt_t = abstract_flt_t.view(n, c , w, h)
    #symbolic_flt_t = symbolic_flt_t.view(n, c , -1)
    #print(abstract_flt_t.shape)
    return abstract_flt_t.float(), buffer_sym 

##dist_lut_768 = np.genfromtxt('./dist_matrix_768_PlatformConv.txt', delimiter=',')
#dist_lut_768 = np.genfromtxt('./dist_matrix_768_8_PlatformConv.txt', delimiter=',')
#
#
#def conv2DSym(fm_prev , filters, biases, kernel_size, stride, # conv parameters 
#              n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count, # patch params
#              lut): # distance params for new conv op
#    # The shape is n, c, w, h 
#    # We need to do some manipulation
#    ana=True 
#    multi=False
#    instr=False
#    pdf=None
#    #print(fm_prev.shape)
#    # Only batch size of 1 supported till now
#    if len(fm_prev.shape) > 2:
#        c, w, h = fm_prev.shape
#    else:
#        c = 1
#        w, h = fm_prev.shape
#    
#    num_output_filters = len(filters)
#    
#    #print(num_output_filters)
#    #print(len(filters[1]))
#
#    #output_fm = np.zeros((((w - kernel_size[0])/stride)+1 ,((h - kernel_size[0])/stride)+1, num_output_filters))
#    symbol_distance = None
#    
#    # First extract and store all patches from all channels of the image
#    fm_prev_sym_array = []
#    for ch in range(c):
#        if c > 1:
#            fm_prev_squeezed = fm_prev[ch,:,:]
#            #print(fm_prev_squeezed.shape)
#            #fm_prev_squeezed = np.squeeze(fm_prev_squeezed)
#            #fm_prev_squeezed = np.expand_dims(fm_prev_squeezed, axis=0)
#            #print(fm_prev_squeezed.shape)
#        else:
#            fm_prev_squeezed = fm_prev
#        #print(fm_prev_squeezed.shape)
#    
#        _, fm_prev_sym = extract_patches_conv(fm_prev_squeezed, (5, 5), 
#                         1, n_clusters, index, 
#                         centroid_lut,  patch_size, patch_stride, 
#                         1)
#        num_patches_for_conv = len(fm_prev_sym) #/ len(fm_prev_sym[0])
#        fm_prev_sym_array.append(fm_prev_sym)
#    
#    num_symbols_within_filter = len(filters[0][0])
#    #print(num_symbols_within_filter)
#    # Find height / width of output FM
#    output_wh = int((w - kernel_size[0])/stride) + 1
#    
##     # create the bias term
##     biases = [ 0.2311,  0.1832, -0.0486,  0.1120,  0.0441,  0.1930]
#    
#    
#    # Create the placeholder for the output feature map (output_channels x height x width)
#    output_fm = np.zeros((num_output_filters, output_wh, output_wh))
#    symbol_distance = 0
#    for k in range(num_output_filters):    
#        for i in range(num_patches_for_conv):
#            for ch in range(c):
#                #symdiff = list(set(fm_prev_sym_array[ch][i]).symmetric_difference(set(filters[k][ch])))
#                #set_diff = len(symdiff)
#                #print(set_diff)
#                for j in range(num_symbols_within_filter):
#                    # Now perform no convolution
#                    #print(len(fm_prev_sym_array),len(fm_prev_sym_array[0]),len(fm_prev_sym_array[0][0]))
#                    #print(fm_prev_sym_array,fm_prev_sym_array[0],fm_prev_sym_array[0][0])
#                    #print(len(filters),len(filters[0]),len(filters[0][0]))
#                    fm_symbol = fm_prev_sym_array[ch][i][j]  # symbols from a kernel_size x kernel_size chunk 
#                    flt_symbol = filters[k][ch][j]
#                    # Find set difference
#                    #print(i, k, j, fm_symbol, flt_symbol, ch)
#                    symbol_distance += (lut[fm_symbol][flt_symbol])
#                    #symbol_distance += (lut[fm_symbol][flt_symbol])/set_diff
#                    #print(lut[fm_symbol][flt_symbol])
#                #print("Done for one channel: ", ch)
#            #print("Done for all channel of one region: ", i)
#            # Get the output row
#            oh = int(i//output_wh)
#            # Get the output column
#            ow = int(i % output_wh)
#            output_fm[k][oh][ow] = symbol_distance + biases[k]
#            symbol_distance = 0
#        #print("Done for all regions")
#    #print("Done for all filters")
#    # Convert to torch and reshape as filter
#    output_fm = torch.from_numpy(output_fm)
#    #output_fm = (output_fm - actual_min)/(actual_max - actual_min)*(desired_max - desired_min) + desired_min
#    output_fm = output_fm.unsqueeze(0)
#    return output_fm

# Basic definitions for MNIST inference

#  From my training code
random_seed = 1 


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
apply_transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                      transforms.Normalize((0.1309,), (0.2893,))])
# Change the dataset folder to the proper location in a new system
testset = datasets.MNIST(root='../../dataset', train=False, download=False, transform=apply_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

random_indices = list(range(0, len(testset), 200))
testset_subset = torch.utils.data.Subset(testset, random_indices)
testloader_subset = torch.utils.data.DataLoader(testset_subset, batch_size=1, shuffle=False)


random_indices_1 = list(range(0, len(testset), 100))
testset_subset_1 = torch.utils.data.Subset(testset, random_indices_1)
testloader_subset_1 = torch.utils.data.DataLoader(testset_subset_1, batch_size=1, shuffle=False)



# apply_transform_nonorm = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])
# # Change the dataset folder to the proper location in a new system
# testset_nonorm = datasets.MNIST(root='../../dataset', train=False, download=True, transform=apply_transform_nonorm)
# random_indices_nonorm = list(range(0, len(testset_nonorm), 200))
# testset_subset_nonorm = torch.utils.data.Subset(testset_nonorm, random_indices)
# testloader_subset_nonorm = torch.utils.data.DataLoader(testset_subset_nonorm, batch_size=1, shuffle=False)

class CNN_LeNet(nn.Module):
    def __init__(self):
        super(CNN_LeNet, self).__init__()
        # Define the net structure
        # This is the input layer first Convolution
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10) 
    
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x


    def forward_symbolic(self, x, n_clusters, index, centroid_lut, patch_size, stride, channel_count): 
        channel_count = 1
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv1(x))
        channel_count = 6
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool1(x)
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv2(x))
        channel_count = 16
        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x 
    
    def forward_symbolic_ana(self, x, n_clusters, index, centroid_lut, patch_size, stride, channel_count): 
        channel_count = 1
        x, x_sym_input = abstract_input_ana(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv1(x))
        channel_count = 6
        x, x_sym_c1 = abstract_input_ana(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool1(x)
        x, x_sym_c2 = abstract_input_ana(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = F.relu(self.conv2(x))
        channel_count = 16
        x, x_sym_c3 = abstract_input_ana(x, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x, x_sym_input, x_sym_c1, x_sym_c2, x_sym_c3  
    
#    def forward_symbolic_noconv1(self, x, filter_set, biases, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count): 
#        channel_count = 1
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        #x = F.relu(self.conv1(x))  # this goes off
#        # Set conv 1 layer parameters
#        kernel_size = (5,5)
#        stride = 1
#        x = F.relu(conv2DSym(x.squeeze() , filter_set, biases , kernel_size, stride, # conv parameters 
#              n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count, # patch params
#              dist_lut_768))
#        channel_count = 6
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool1(x)
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = F.relu(self.conv2(x))
#        channel_count = 16
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool2(x)
#        x = x.view(-1, 400)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        x = F.softmax(x,dim=1)
#        return x   
#    def forward_symbolic_noconv2(self, x, filter_set, biases, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count): 
#        channel_count = 1
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = F.relu(self.conv1(x))  
#        channel_count = 6
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool1(x)
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        # Set conv 2 layer parameters
#        kernel_size = (5,5)
#        stride = 1
#        x = F.relu(conv2DSym(x.squeeze() , filter_set, biases, kernel_size, stride, # conv parameters 
#              n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count, # patch params
#              dist_lut_768))
#        
#        #x = F.relu(self.conv2(x)) This goes off
#        channel_count = 16
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool2(x)
#        x = x.view(-1, 400)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        x = F.softmax(x,dim=1)
#        return x   
#    def forward_symbolic_noconv_all(self, x, filter_sets, biasess, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count): 
#        channel_count = 1
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        #x = F.relu(self.conv1(x))  # this goes off
#        # Set conv 1 layer parameters
#        kernel_size = (5,5)
#        stride = 1
#        convid = 'c1' 
#        x = F.relu(conv2DSym(x.squeeze() , filter_sets[convid], biasess[convid], kernel_size, stride, # conv parameters 
#              n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count, # patch params
#              dist_lut_768))
#        channel_count = 6
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool1(x)
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        #x = F.relu(self.conv2(x)) # This goes off
#        # Set conv 2 layer parameters
#        kernel_size = (5,5)
#        stride = 1
#        convid = 'c2' 
#        x = F.relu(conv2DSym(x.squeeze() , filter_sets[convid], biasess[convid], kernel_size, stride, # conv parameters 
#              n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count, # patch params
#              dist_lut_768))
#        channel_count = 16
#        x = abstract_input(x, n_clusters, index, centroid_lut, patch_size, patch_stride, channel_count)
#        x = self.pool2(x)
#        x = x.view(-1, 400)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        x = F.softmax(x,dim=1)
#        return x   
#

# Test accuracy of symbolic inference
def mnist_test_fullsym_acc(model, atk, filter_set , biases, data_iter, clamp, conv, n_clusters, index,  patch_size, stride, channel_count):
    correct = 0 
    total = 0 
    centroid_lut = index.reconstruct_n(0, n_clusters)
    model.eval()
    with torch.no_grad():
        for data in data_iter:
            X, y = data
            if(clamp):
                X = softclamp01(X)
            if atk:
                X_atk = atk(X, y)
                X = X_atk
            if conv == 100:
                output = model(X)
            elif conv == 0:
                output = model.forward_symbolic(X,  n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            elif conv == 1:
                output = model.forward_symbolic_noconv1(X, filter_set, biases, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            elif conv == 2:
                output = model.forward_symbolic_noconv2(X, filter_set, biases, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            else:
                output = model.forward_symbolic_noconv_all(X, filter_set, biases, n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    return round(correct/total, 4)


pretrained_sc_model = "./mnist_sc_v0_best.pt"
#Time to patch a model with the new filters
mnist_model = None
mnist_model = CNN_LeNet() 
mnist_model.load_state_dict(torch.load(pretrained_sc_model))
mnist_model.eval()

pretrained_sc_model = "./mnist_sc_v0_best.pt"
#Time to patch a model with the new filters
mnist_sym_model_stdflt = None
mnist_sym_model_stdflt = CNN_LeNet() 
mnist_sym_model_stdflt.load_state_dict(torch.load(pretrained_sc_model))
mnist_sym_model_stdflt.eval()


# Detect the cases
def mnist_detect_cases(model, atk, filter_set , biases, data_iter, clamp, conv, n_clusters, index,  patch_size, stride, channel_count):
    correct_o = False 
    correct_s = False
    correct_oa= False
    correct_sa = False
 
    centroid_lut = index.reconstruct_n(0, n_clusters)
    model.eval()
    cases = []
    counter = 0
    with torch.no_grad():
        for data in data_iter:
            correct_o = False 
            correct_s = False 
            correct_oa= False
            correct_sa = False
            X, y = data
            if(clamp):
                X = softclamp01(X)
            X_o = X    
            if atk:
                X_atk = atk(X, y)
                X = X_atk
            
            # check cases one by one
            # original
            output_o = model(X_o)
            output_o = output_o.squeeze()
            
            if torch.argmax(output_o) == y:
                #print("correct original")
                correct_o = True
            
            # symbolic 
            output_s = model.forward_symbolic(X_o,  n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            output_s = output_s.squeeze()
            
            if torch.argmax(output_s) == y:
                #print("correct symbolic")
                correct_s = True

            # original attacked
            output_oa = model(X)
            output_oa = output_oa.squeeze()
            
            if torch.argmax(output_oa) == y:
                #print("correct original attacked")
                correct_oa = True

            # symbolic attacked 
            output_sa = model.forward_symbolic(X,  n_clusters, index, centroid_lut, patch_size, stride, channel_count)
            output_sa = output_sa.squeeze()
            
            if torch.argmax(output_s) == y:
                #print("correct symbolic")
                correct_sa = True

            # Now the cases 
            if correct_o:
                if correct_s:
                    if correct_oa:
                        if correct_sa:
                            print(counter," : ",1)
                        else:
                            print(counter," : ",2)
                    else:          
                        if correct_sa:
                            print(counter," : ",3)
                        else:
                            print(counter," : ",4)
                else:
                    print(counter," : ",5)
            else:
                print(counter," : ",6)

            counter +=1 


#cases_0 = mnist_detect_cases(mnist_sym_model_stdflt, None, None, None, testloader, True , 0, n_clusters, index,  patch_size, patch_stride, channel_count)
# print("Detection done - No attack")
# print(cases_0)
import torchvision.utils
from torchvision import models
import torchattacks
from torchattacks import *

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
attack = AutoAttack(mnist_sym_model_stdflt, eps=16/255, n_classes=10, version='standard')
cases_16 = mnist_detect_cases(mnist_sym_model_stdflt, attack, None, None, testloader, True , 0, n_clusters, index,  patch_size, patch_stride, channel_count)
print("Detection done - 16 attack")
print(cases_16)

# attack = AutoAttack(mnist_sym_model_stdflt, eps=8/255, n_classes=10, version='standard')
# cases_8 = mnist_detect_cases(mnist_sym_model_stdflt, attack, None, None, testloader, True , 0, n_clusters, index,  patch_size, patch_stride, channel_count)
# print("Detection done - 8 attack")
# print(cases_8)


# np.savetxt('cases_16.txt', cases_16, delimiter =', ', fmt='%d')
# #np.savetxt('cases_8.txt', cases_8, delimiter =', ', fmt='%d')
