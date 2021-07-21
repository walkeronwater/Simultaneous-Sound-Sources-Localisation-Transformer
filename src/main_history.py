import argparse
import time
import soundfile as sf
from scipy import signal
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import h5py
import csv
import pandas as pd
from glob import glob
import librosa
import librosa.display
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

from load_data import *
from utils import *
from model_transformer import *
from loss import DoALoss
from main_train import loadCheckpoint

def loadHistory(loadPath, figPath, isDebug):
    trainHistory = glob(os.path.join(loadPath, "curve*"))

    history = {
        'train_acc': [],
        'train_loss': [],
        'valid_acc': [],
        'valid_loss': []
    }
    for i in range(len(trainHistory)):
        checkpt = torch.load(
            loadPath+"curve_epoch_"+str(i+1)+".pth.tar"
        )
        for idx in history.keys():
            history[idx].append(round(checkpt[idx], 5))
    if isDebug:
        for idx in history.keys(): 
            print(idx, history[idx])

    # plt.subplot(211, figs)
    # fig = matplotlib.pyplot.gcf()
    # figure(figsize=(8, 6), dpi=80)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'])
    plt.plot(range(1, len(history['train_loss'])+1), history['valid_loss'])
    plt.xlabel('epoch')
    plt.xticks(range(1, len(history['train_loss'])+1))
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.savefig(figPath+"loss.png")
    plt.show()

    # plt.subplot(212)
    plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'])
    plt.plot(range(1, len(history['train_acc'])+1), history['valid_acc'])
    plt.xlabel('epoch')
    plt.xticks(range(1, len(history['train_acc'])+1))
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.savefig(figPath+"acc.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training history plot')
    parser.add_argument('modelDir', type=str, help='Directory of model')
    parser.add_argument('--figDir', type=str, help='Directory of figures to be saved at')
    parser.add_argument('--isDebug', type=str, default="False", help='isDebug?')

    args = parser.parse_args()
    if args.modelDir[-1] != "/":
        args.modelDir += "/"
    if args.figDir == None:
        args.figDir = args.modelDir
    else:
        if args.figDir[-1] != "/":
            args.figDir += "/"
    if args.isDebug == "True":
        args.isDebug = True
    else:
        args.isDebug = False
    
    print("Model directory: ", args.modelDir)
    
    loadHistory(args.modelDir, args.figDir, args.isDebug)