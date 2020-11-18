# Run Time Adaptive Network Slimming for Mobile Environments

#### Authors: [Hong-Ming, Chiu](https://hong-ming.github.io/), [Kuan-Chih Lin](), [Tian Sheuan Chang](https://eenctu.nctu.edu.tw/tw/teacher/p1.php?num=108&page=1)
#### [Link to Paper](https://ieeexplore.ieee.org/document/8701884)
Code for training resnet models referenced from: [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
#### Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{chiu2019,
  author={Hong-Ming Chiu and Kuan-Chih Lin and Tian Sheuan Chang,
  booktitle={2019 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
  title={Run Time Adaptive Network Slimming for Mobile Environments}, 
  year={2019},
  pages={1-4},
  doi={10.1109/ISCAS.2019.8701884}}
```

## Intorduction
This Python program contains the code for the paper "Run Time Adaptive Network Slimming for Mobile Environments" and the code for training the ResNet model. This program performs the interface stage pruning and conculates the pruning rate based on the saving in floating point operations (FLOP).

## Contents
1. In `/`:
    - **Adapted_Network.py**: main function for Adaptive Network Slimming. [1]
2. In `model_pkl/`: 
    - **\<model name>.pkl**: pretrained model file.
3. In `train_model/` [5] : 
    - **main.py**: main function for training model.
    
## Setup/Installation
1. Package Version
    - Python 3.6
    - PyTorch 1.10
    - Torchvision 0.3.0
    - CUDA 10.0
2. Hardware and Operation System
    - CPU: Intel i7-8700 3.2GHz
    - Graphics Cards: GeForce RTX 2080 Ti
    - OS: Ubuntu 18.04
3. Setup the environment using **Anaconda** 
   ```
   conda create -n myenv python=3.6
   conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
   ```
3. This program stores the outputs and learned parameters in `Data/` to speed up implementation, run **cleanup** to cleanup those data
   ```
   >>conda create -n myenv python=3.6
   ```
        
## Reference
[1] Hong-Ming Chiu, Kuan-Chih Lin and Tian Sheuan Chang, “Run Time Adaptive Network Slimming for Mobile Environments,” 2019 IEEE International Symposium on Circuits and Systems (ISCAS).
    
[2] Ferris Brian, Hähnel Dirk and Fox Dieter, "Gaussian Processes for Signal Strength-Based Location Estimation," 2006.
    
[3] A. Serrano, B. Girault, and A. Ortega, "Geostatistical Data Interpolation using Graph Signal Spectral Prior," 2019.
    
[4] R. Di Taranto et al., "Location-aware communicationsfor 5g networks," IEEE Signal Processing Magazine, vol. 31, no. 6, pp. 102?112, 2014.
    
[5] H.E. Egilmez, E. Pavez, and A. Ortega, "Graph learn-ing from data under laplacian and structural constraints," IEEE Journal of Selected Topics in Signal processing, vol. 11, no. 6, pp. 825?841, 2017.
    
[6] Michael Grant and Stephen Boyd. CVX: Matlab software for disciplined convex programming, version 2.0 beta. http://cvxr.com/cvx, September 2013.
    
## Author/Conatact Info
Name  : Hong-Ming, Chiu

Email : hongmingchiu2017@gmail.com
