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

from utils_train import *
from model_transformer import DecoderFC

class CNNModel(nn.Module):
    def __init__(self, task, Ncues, dropout, device, isDebug=False):
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
            # nn.Linear(2560, 1024),
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
        self.device = device
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

class CNN_multiSound(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        dropout,
        device,
        isDebug=False
    ):
        super(CNN_multiSound, self).__init__()

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
        self.FClayers_elev = nn.Sequential(
            # nn.Linear(7872, 256),
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound, bias=False)
        )

        self.FClayers_azim = nn.Sequential(
            # nn.Linear(7872, 256),
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound, bias=False)
        )

        self.decoder_FC = DecoderFC(
            task=task,
            flattenShape=7872,
            Nsound=Nsound,
            dropout=dropout,
            device=device
        )

        self.device = device
        self.isDebug = isDebug
    def forward(self, cues):
        out = self.convLayers(cues.permute(0,3,2,1))
        if self.isDebug:
            print("Shape after convLayers: ", out.shape)
        out = torch.flatten(out, 1, -1)

        if self.isDebug:
            print("Shape after flatten: ", out.shape)

        return self.decoder_FC(out)
# default uniform method provided by Pytorch is Kaiming He uniform
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    numEnc = 6
    numFC = 3
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    Nsound = 2
    batchSize = 32
    dropout = 0
    # model = CNNModel(task=task, Ncues=Ncues, dropout=dropout, device=device, isDebug=True).to(device)
    model = CNN_multiSound(
        task=task,
        Ntime=Ntime,
        Nfreq=Nfreq,
        Ncues=Ncues,
        Nsound=Nsound,
        dropout=dropout,
        device=device,
        isDebug=False
    ).to(device)
    model.apply(weight_init)

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
    