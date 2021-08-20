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

from utils import *
from utils_train import *

class CRNN(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        whichDec,
        num_conv_layers=4,
        num_recur_layers=2,
        device="cpu",
        dropout=0.1,
        isDebug=False,
        coordinates="spherical"
    ):
        super(CRNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(Ncues, 64, (3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1,4))
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2))
        )

        self.conv_layers = nn.ModuleList(
            [
                self.conv_blocks
                for _ in range(num_conv_layers)
            ]
        )

        self.recur_layers = nn.LSTM(
            input_size=256, hidden_size=128, bidirectional=True,
            dropout=dropout,
            num_layers=num_recur_layers,
            batch_first=True
        )
        
        self.FC_blocks_elev = nn.Sequential(
            nn.Linear(52*256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2, bias=False)
        )

        self.FC_blocks_azim = nn.Sequential(
            nn.Linear(52*256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2, bias=False)
        )

        self.isDebug = isDebug

    def forward(self, inputs):
        # shape after permutation: 
        inputs = inputs.permute(0, 3, 2, 1)
        if self.isDebug: print(f"After input permutation: {inputs.shape}")

        out = self.conv_1(inputs)
        for layers in self.conv_layers:
            out = layers(out)
        
        out = out.permute(0, 2, 1, 3)
        if self.isDebug: print(f"CNN output: {out.shape}")

        out = torch.flatten(out, 2, -1)
        if self.isDebug: print(f"Flattened output: {out.shape}")

        out, (hn, cn) = self.recur_layers(out, None)
        if self.isDebug: print(f"RNN output: {out.shape}")

        out = torch.flatten(out, 1, -1)
        out_elev = self.FC_blocks_elev(out)
        out_azim = self.FC_blocks_azim(out)

        out = torch.cat([out_elev, out_azim], dim=-1)

        return out

if __name__ == "__main__":
    # rnn = nn.LSTM(
    #     input_size=256, hidden_size=128, num_layers=2,
    #     bidirectional=True,
    #     # dropout=
    #     batch_first=True
    # )
    # inputs = torch.randn(32, 52, 256)
    # h0 = torch.randn(4, 32, 128)
    # c0 = torch.randn(4, 32, 128)
    # output, (hn, cn) = rnn(inputs, (h0, c0))
    # output, (hn, cn) = rnn(inputs, None)
    # print(output.shape)

    # raise SystemExit

    path = "./HRTF/IRC*"
    _, locLabel, _ = loadHRIR(path)

    path = "./saved_0808_temp/"
    csvF = pd.read_csv(path+"/train/dataLabels.csv", header=None)


    temp_1 = torch.load(path+"/train/0.pt")
    temp_2 = torch.load(path+"/train/1.pt")
    cues_tensor = torch.stack([temp_1, temp_2], dim=0)
    print(cues_tensor.shape)
    # define
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    Nsound = 2
    batch_size = 32
    num_workers = 0
    isPersistent = True if num_workers > 0 else False

    train_dataset = CuesDataset(path + "/train/",
                                task, Nsound, locLabel, isDebug=False)
    train_loader = MultiEpochsDataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=isPersistent
        )
    Nfreq = train_dataset.Nfreq
    Ntime = train_dataset.Ntime
    Ncues = train_dataset.Ncues
    print(f"Nfreq: {Nfreq}, Ntime: {Ntime}, Ncues: {Ncues}")

    model = CRNN(
        task=task,
        Ntime=Ntime,
        Nfreq=Nfreq,
        Ncues=Ncues,
        Nsound=Nsound,
        whichDec="src",
        num_conv_layers=4,
        num_recur_layers=2,
        dropout=0.1,
        device=device,
        isDebug=False,
        coordinates="spherical"
    )
    model = model.to(device)

    inputs, labels = next(iter(train_loader))
    outputs = model(inputs.to(device))

    print(f"inputs: {inputs.shape},\
        max outputs: {torch.max(outputs)}, \
        labels: {labels.shape}")