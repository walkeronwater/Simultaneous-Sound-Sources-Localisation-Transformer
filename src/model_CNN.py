import soundfile as sf
from scipy import signal
import random
from math import pi
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

from utils_model import *

class CNNModel(nn.Module):
    def __init__(self, task, Ncues, dropout, isDebug=False):
        super(CNNModel, self).__init__()

        self.task = task
        Nloc = predNeuron(task)
        self.convLayers = nn.Sequential(
            nn.Conv2d(Ncues, 32, (5,5), stride=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3,3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, (3,3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            # nn.Conv2d(96, 128, (2,2), stride=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(128)
        )
        self.FCLayers = nn.Sequential(
            nn.Linear(7872, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, Nloc)
        )
        if task in ["elevRegression","azimRegression","allRegression"]:
            self.setRange1 = nn.Hardtanh()
            self.setRange2 = nn.Hardtanh()
        self.isDebug = isDebug
    def forward(self, cues):
        out = self.convLayers(cues.permute(0,3,2,1))
        if self.isDebug:
            print("Shape after convLayers: ", out.shape)
        out = torch.flatten(out, 1, -1)

        if self.isDebug:
            print("Shape after flatten: ", out.shape)

        out = self.FCLayers(out)
        if self.isDebug:
            print("Shape after FClayers (output): ", out.shape)

        if self.task in ["elevRegression","azimRegression","allRegression"]:
            out = torch.stack(
                [
                    3/8*pi*self.setRange1(out[:,0])+pi/8,
                    pi*self.setRange2(out[:,1])+pi
                ], dim=1
            )
        return out
    
    # default uniform method provided by Pytorch is Kaiming He uniform
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    #         nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    numEnc = 6
    numFC = 3
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    batchSize = 32
    model = CNNModel(task=task, Ncues=Ncues, dropout=0, isDebug=True).to(device)

    testInput = torch.rand(batchSize, Nfreq, Ntime, Ncues, dtype=torch.float32).to(device)
    print("testInput shape: ", testInput.shape)
    # print(testLabel)

    # print(testInput.permute(0,3,1,2).shape)
    # raise SystemExit("debug")
    testOutput = model(testInput)
    print("testOutput shape: ",testOutput.shape)
    # print("testOutput: ",testOutput)
    # print(torch.max(testOutput, 1))

    summary(model, (Nfreq, Ntime, Ncues))
    