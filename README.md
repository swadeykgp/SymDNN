# SymDNN: Simple Effective Adversarial Robustness for Embedded Systems


SymDNN is a Python and PyTorch based Deep Neural Network Inference scheme library that provides *defense* against *adversarial attacks* and results in image compression.  

<details><summary>Quick Usage Guide</summary><p>

```python
# For similarity search
import faiss
import sys
sys.path.insert(1, './core')

# Import the main function for purification of adversarial perturbation
from patchutils import symdnn_purify

# Setup some similarity search parameters and select the desired index
channel_count = 3
stride = 0
n_clusters = 2048
patch_size = (2, 2)
location=False
index = faiss.read_index('./cifar10/kmeans_img_k2_s0_c2048_v1_softclamp.index')
centroid_lut = index.reconstruct_n(0, n_clusters)

# Start purifying adversarial images and use that for further DNN inference
purified_image = symdnn_purify(attacked_image, n_clusters, index, centroid_lut, patch_size, stride, channel_count)


```
</p></details>



## Table of Contents

1. [Setup](#Requirements-and-Installation)
2. [Examples and Result Reproduction](#Examples-and-Result-Reproduction)


## Setup

- Step 1. Create basic virtual env:

```
sudo apt install python3.8-venv
python3 -m venv symbol
source symbol/bin/activate
pip install  -r symdnn_base.txt

```


## Examples and Result Reproduction

### Compaction

1. Reproduce the compaction (Table 3 of paper): run: python imagenet_compaction_table_3_reproduce.py ( for this to work please put the ImageNet dataset in a folder with train & val directories under it. Point that location in the python file) . For Table 1 entropy encoding can be tested by:
        1. cd imagenet
        2. Open Jupyter notebook
        3. Run: imagenet-symbolic-inference-size-experiments.ipynb
2. MNIST compaction can be reproduced as follows:
        1. cd mnist
        2. Open Jupyter notebook
        3. Run: mnist-symbolic-entropy-encoding.ipynb
3. CIFAR10 compaction can be reproduced as follows:
        1. cd cifar10
        2. Open Jupyter notebook
        3. Run: cifar-symbolic-entropy-encoding.ipynb

### Adversarial Robustness

Table 1 results (CIFAR-10 Adversarial Robustness under different Attack models) can be produced as follows:

#### Model White box / Defence Black Box - SymDNN:

```
python cifar10_modelwb_defensebb_table1.py
```


#### Model Black box / Defence Black Box - SymDNN:

```
python cifar10_modelbb_defensebb_table1.py
```

- Note: The given holdout model can be replaced under cifar10 directory

#### Model White box / Defence White box - SymDNN:

```
python cifar10_modelwb_defensewb_table1.py
```

- Note: This is the gradient obfuscation case, check how SymDNN defends BPDA


#### Model White box / Defence Black Box - Comparison with some State-of-the-Art defenses:

Table 2 results have to be reproduced in four phases - SymDNN Model White box / Defence Black Box is already obtained in step 4. For NRP we provide the NRP directory with SymDNN files. Go to the NRP directory under cifar10. Download the pretrained purifier models for NRP - NRP.pth ( Please check their repository: https://github.com/Muzammal-Naseer/NRP)
##### NRP
```
cd cifar10/NRP/
python cifar10_modelwb_defensebb_table2.py
```
##### DefenseGAN
We supply the working repo with modifications for testing under all TorchAttacks attacks. If one uses the latest repo from : (https://github.com/sky4689524/DefenseGAN-Pytorch), the following changes are required:

    gen_cifar10_gp_4999.pth provided under defensive_models/

        1. in util_defense_GAN.py:
        line 55
        - reconstruct_loss.backward()
        + reconstruct_loss.backward(retain_graph=True)

        2. ```python cifar10_modelwb_defensebb_table2.py```

##### Adversarially Trained models
- Note: Install Robustbench version 0.1 for this part

```
python cifar10_test_adversarially_trained_models.py
```

#### Preliminary experiments on ImageNet:

For testing ImageNet adversarial attacks install Foolbox version 2.3.0:

```    
pip install foolbox==2.3.0
cd imagenet
```

1. Open Jupyter notebook
2. Run: imagenetviz.ipynb

### Ablation and Other Experiments

6. The comparison of edit distance and L2 distance of C&W attack, presented in Sec. 5.4 can be reproduced as follows:
        1. run : python figure3_reproduce.py
        2. Move the generated files - l2_distances*.txt , edit_distances*.txt to cifar10/cw_dist_case3 directory
        3.  Open Jupyter notebook
        4.  Run: inspecting_cw_attack.ipynb

7. The experiments related to the SymDNN inference using the common imagenet codebook is presented in the mnist/section-5.4-mnist-with-ImageNet-index.ipynb  and cifar10/section-5.4-cifar-with-ImageNet-index.ipynb
8. Most of the experiments and visualization in the supplementary material are available in the ipynb files in the respective directories. For instance the centroid separation plot can be generated using mnist/mnist-centroid-distance.ipynb
9. The clustering of ImageNet and MNIST can be tried out in the clustering folder. For MNIST cd to clustering/mnist and run: python mnistcluster.py
10. For Imagenet, we recommend a GPU system. Please create a new python venv and run: pip install -r symdnn_cluster_gpu.txt  .  Then cd to   clustering/imagenet and run: python imgnet_cluster.py






