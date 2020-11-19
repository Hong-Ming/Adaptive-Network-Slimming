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
  doi={10.1109/ISCAS.2019.8701884}
  }
```
## Table of Contents
* [Intorduction](#intorduction)
* [Contents](#contents)
* [Setup/Installation](setup/installation)

## Intorduction
This Python program contains the code for the paper "Run Time Adaptive Network Slimming for Mobile Environments" and the code for training the ResNet [2] model. This program performs the interface stage pruning and conculates the pruning rate based on the saving in floating point operations (FLOP).

## Contents
1. In `/`:
    - **Adapted_Network.py**: main function for Adaptive Network Slimming. [1]
2. In `model_pkl/`: 
    - **\<model name>.pkl**: pretrained model file.
3. In `train_model/`
    - **main.py**: main function for training model.
    
## Setup/Installation
### Package Version
- Python 3.6
- PyTorch 1.10
- Torchvision 0.3.0
- CUDA 10.0
### Hardware and Operation System
- CPU: Intel i7-8700 3.2GHz
- Graphics Cards: GeForce RTX 2080 Ti
- OS: Ubuntu 18.04
### Environment setup
1. Set up the environment using [Anaconda](https://www.anaconda.com/)
```sh
conda create -n myenv python=3.6
conda install --name myenv pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```
2. Activate conda environment
```sh
conda activate myenv
```
3. Run network slimming model
```sh
python3 Adapted_Network.py
```
        
## Reference
[1] Hong-Ming Chiu, Kuan-Chih Lin and Tian Sheuan Chang, "Run Time Adaptive Network Slimming for Mobile Environments," 2019 IEEE International Symposium on Circuits and Systems (ISCAS).

[2] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun, "Deep Residual Learning for Image Recognition," 2015.
    
## Author/Conatact Info
Name  : Hong-Ming, Chiu

Email : hongmingchiu2017@gmail.com
