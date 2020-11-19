
import sys
sys.path.append('./train_model')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='Adaptive Network Slimming')
parser.add_argument('-net', type=str, help='pretrained pkl file')
parser.add_argument('--nonuniform', action='store_true', help='set non-uniform pruning rate')
args = parser.parse_args()

# from models import *

transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

testset = torchvision.datasets.CIFAR10(root='./cifar10',train=False,download=True,transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=128,shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

input_shape = (3,32,32)
if  args.net == "resnet18":
    START = 0.2
    END = 0.81
    netnum = 18
elif args.net == "resnet34":
    START = 0.2
    END = 0.81
    netnum = 34
elif args.net == "resnet50":
    START = 0.2
    END = 0.8
    netnum = 50
elif args.net == "resnet101":
    START = 0.2
    END = 0.8
    netnum = 101
elif args.net == "resnet152":
    START = 0.21
    END = 0.79
    netnum = 152

if args.nonuniform:
    PRUNE_RATE = np.arange(START,END,(END-START)/(netnum-1))
    FC_PRUNE_RATE = END
    Model_Name = "ResNet" + str(netnum) + " (Non-uniform Pruning Rate)"
else:
    PRUNE_RATE = np.zeros([netnum-1,1]) + 0.5
    FC_PRUNE_RATE = 0.5
    Model_Name = "ResNet" + str(netnum) + " (Uniform Pruning Rate)"

# -------------- Load Pretrained Model---------------
File_Name = "./model_pkl/" + args.net + ".pkl"
net = torch.load(File_Name, map_location= "cpu")

def RunData():
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        net.cuda()
        for (x,y) in testloader:
            xa = x.cuda()
            ya = y.cuda()
            out = net(xa)
            _,predicted = torch.max(out.data,1)
            total += y.size(0)
            correct += (predicted.cpu() == y).sum()
    net.cpu()
    Accuracy = 100*correct.cpu().numpy()/total
    return Accuracy

def RunData2():
    correct = 0
    total = 0
    for _,layer in net.named_modules():
        if isinstance(layer,nn.BatchNorm2d):
            layer.track_running_stats=False

    with torch.no_grad():
        net.eval()
        net.cuda()
        for (x,y) in testloader:
            xa = x.cuda()
            ya = y.cuda()
            out = net(xa)
            _,predicted = torch.max(out.data,1)
            total += y.size(0)
            correct += (predicted.cpu() == y).sum()
        net.cpu()
        Accuracy = 100*correct.cpu().numpy()/total
    return Accuracy

def prune_filter(layer,PRUNE_RATE):
    prune = np.sum(abs(layer),axis = (1,2,3))
    sort_prune = np.sort(prune)
    mask = np.ones(layer.shape)
    for i in range(len(prune)):
        if prune[i] < sort_prune[int(np.floor(PRUNE_RATE*len(prune)))]:
            mask[i,:] = 0
    return mask

def prune_weight(layer,PRUNE_RATE):
    layer_flatten_sort = np.sort(abs(layer.flatten()))
    mask = np.ones(layer.shape)
    for i in range(layer.shape[0]):
        for j in range(layer.shape[1]):
            if abs(layer[i][j]) < layer_flatten_sort[int(np.floor(PRUNE_RATE*len(layer_flatten_sort)))]:
                mask[i][j] = 0
    return mask

def Calculate_flop():
    FLOP = 0
    shape = input_shape[1]
    for name,layer in net.named_modules():
        if isinstance(layer,nn.Conv2d) and 'shortcut' not in name:
            filter_data = layer.weight.data.numpy()
            skip = sum(np.sum(abs(filter_data),axis = (1,2,3)) == 0)
            filter_shape = layer.weight.data.numpy().shape
            padding = layer.padding
            stride = layer.stride
            n = filter_shape[1] * filter_shape[2] * filter_shape[3] # vector length
            fpn = n + (n - 1)     # n multiplication, n-1 addition
            step_x = np.floor(((shape - filter_shape[2] + 2 * padding[0]) / stride[0]) + 1)
            shape = step_x
            step = step_x**2
            fpf = fpn*step
            FLOP += fpf*(filter_shape[0] - skip)
    
        elif isinstance(layer,nn.Linear):
            filter_data = layer.weight.data.numpy()
            skip = sum(sum(filter_data == 0))
            filter_shape = layer.weight.data.numpy().shape
            FLOP += 2 * (filter_shape[0] * filter_shape[1] - skip)
    return FLOP

ACC_before = RunData()
print("Model Name: " + Model_Name)
print("Accuracy  : " + str(ACC_before) + "%")
FLOP_before = Calculate_flop()
if FLOP_before / 1e9 > 1:   # for Giga Flops
    print("FLOP      : %4.2f GFLOP" % (FLOP_before / 1e9))
else:
    print("FLOP      : %4.2f MFLOP" % (FLOP_before / 1e6))

print("                                                   ")
print("                   Start Pruning                   ")
print("---------------------------------------------------")
print("|Layer|  FLOP  |#Filter or #Weight|Pruning |Filter|")
print("| No. |  Save  |   before/after   | Type   | Size |")
print("|-----|--------|------------------|--------|------|")

# pruning
TOTAL_WEIGHT = 0
PRUNE_WEIGHT = 0
i = 0
for parname,layer in net.named_modules():

    if isinstance(layer,nn.Conv2d) and 'shortcut' not in parname:
        par = layer.weight.data.numpy()
        par_size = par.shape
  
        mask = prune_filter(par,PRUNE_RATE[i])
        par = (par * mask)
        print("| %3i" % (i+1), "|"+ 
              " %5.2f" % float((1-(np.count_nonzero(mask)/mask.size)) * 100) + "% |"+
              "    %4i" % int((mask.size-np.count_nonzero(mask))/(par_size[1]*par_size[2]*par_size[2])),"/",
              "%4i" % int(mask.size/(par_size[1]*par_size[2]*par_size[2])) + "   | Filter |"+
              " %1ix%1i  |" % (par_size[2], par_size[3]))
        TOTAL_WEIGHT = TOTAL_WEIGHT + (mask.size/(par_size[1]))
        PRUNE_WEIGHT = PRUNE_WEIGHT + ((mask.size-np.count_nonzero(mask))/(par_size[1]))
        i = i + 1
        layer.weight.data = torch.from_numpy(par).type(torch.FloatTensor)

    elif isinstance(layer,nn.Linear):
        par = layer.weight.data.numpy()
        par_size = par.shape
        mask = prune_weight(par,FC_PRUNE_RATE)
        par = (par * mask)
        print("| %3i" % (i+1), "|"+ 
              " %5.2f" % float((1-(np.count_nonzero(mask)/mask.size)) * 100) + "% |"+
              "   %5i" % int(mask.size-np.count_nonzero(mask)),"/",
              "%5i" % int(mask.size) + "  | Weight |" + " none |")
        TOTAL_WEIGHT = TOTAL_WEIGHT + (mask.size)
        PRUNE_WEIGHT = PRUNE_WEIGHT + (mask.size-np.count_nonzero(mask))
        i = i + 1
        layer.weight.data = torch.from_numpy(par).type(torch.FloatTensor)

print("---------------------------------------------------")
ACC_after = RunData2()
FLOP_after = Calculate_flop()
print("                                                   ")
print("                   After Pruning                   ")
print("Accuracy : " + str(ACC_before) + "% -> " + str(ACC_after) + "%")
if FLOP_after / 1e9 > 1:   # for Giga Flops
    if FLOP_before / 1e9 > 1:   # for Giga Flops
        print("FLOP     : %4.2f GFLOP" % (FLOP_before / 1e9) + " -> %4.2f GFLOP" % (FLOP_after / 1e9))
    else:
        print("FLOP     : %4.2f MFLOP" % (FLOP_before / 1e6) + " -> %4.2f GFLOP" % (FLOP_after / 1e9))
else:
    if FLOP_before / 1e9 > 1:   # for Giga Flops
        print("FLOP     : %4.2f GFLOP" % (FLOP_before / 1e9) + " -> %4.2f MFLOP" % (FLOP_after / 1e6))
    else:
        print("FLOP     : %4.2f MFLOP" % (FLOP_before / 1e6) + " -> %4.2f MFLOP" % (FLOP_after / 1e6))

print("FLOP save: %5.2f" % (100*(FLOP_before - FLOP_after)/FLOP_before),"%")