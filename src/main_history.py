import argparse
import time
import soundfile as sf
from scipy import signal
import random
import numpy as np
import matplotlib.pyplot as plt
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

def loadHistory(loadPath):
    trainHistory = glob(os.path.join(loadPath, "curve*"))

    history = {
        'train_acc': [],
        'train_loss': [],
        'valid_acc': [],
        'valid_loss': []
    }
    for i in range(len(trainHistory)):
        checkpt = torch.load(trainHistory[i])
        for idx in history.keys():
            history[idx].append(checkpt[idx])

    plt.plot(history['train_acc'])
    plt.plot(history['valid_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.show()

    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training history plot')
    parser.add_argument('modelDir', type=str, help='Directory of model')

    args = parser.parse_args()
    if args.modelDir[-1] != "/":
        args.modelDir += "/"
    
    print("Model directory: ", args.modelDir)
    
    loadHistory(args.modelDir)