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

class SelfAttention(nn.Module):
    def __init__(self, Nfreq, heads):
        super(SelfAttention, self).__init__()
        self.Nfreq = Nfreq
        self.heads = heads
        self.head_dim = Nfreq // heads

        # assert debug
        assert (
            self.head_dim * heads == Nfreq
        ), "Embedding size needs to be divisible by heads"

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, Nfreq)

    def forward(self, value, key, query):
        # Get number of training examples
        N = query.shape[0]

        value_time, value_freq = value.shape[1], value.shape[2]
        key_time, key_freq = key.shape[1], key.shape[2]
        query_time, query_freq = query.shape[1], query.shape[2]

        # Split the embedding into self.heads different pieces
        value = value.reshape(N, value_time, self.heads, self.head_dim)
        key = key.reshape(N, key_time, self.heads, self.head_dim)
        query = query.reshape(N, query_time, self.heads, self.head_dim)

        values = self.values(value)  # (N, value_len, heads, head_dim)
        keys = self.keys(key)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.Nfreq ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_time, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class Attention(nn.Module):
    def __init__(self, embedSize):
        super(Attention, self).__init__()
        self.embedSize = embedSize

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(embedSize, embedSize, bias=False)
        self.keys = nn.Linear(embedSize, embedSize, bias=False)
        self.queries = nn.Linear(embedSize, embedSize, bias=False)
        self.fc_out = nn.Linear(embedSize, embedSize)

    def forward(self, value, key, query):
        # Get number of training examples
        N = query.shape[0]

        value_time, value_freq = value.shape[1], value.shape[2]
        key_time, key_freq = key.shape[1], key.shape[2]
        query_time, query_freq = query.shape[1], query.shape[2]

        values = self.values(value)  # (N, value_len, heads, head_dim)
        keys = self.keys(key)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("ntqe,ntke->neqk", [queries, keys])

        attention = torch.softmax(energy / (self.embedSize ** (1 / 2)), dim=3)

        out = torch.einsum("neqk,ntve->ntqe", [attention, values]).reshape(
            N, query_time, query_freq, self.embedSize
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, Nfreq, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(Nfreq, heads)
        self.norm1 = nn.LayerNorm(Nfreq)
        self.norm2 = nn.LayerNorm(Nfreq)

        self.feed_forward = nn.Sequential(
            nn.Linear(Nfreq, forward_expansion * Nfreq),
            nn.ReLU(),
            nn.Linear(forward_expansion * Nfreq, Nfreq),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        Nfreq, # frequency bins
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.Nfreq = Nfreq
        self.device = device

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    Nfreq,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Ntime, Nfreq = x.shape[-2], x.shape[-1]
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(x)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)

        return out

class FC3(nn.Module):
    def __init__(
        self,
        task,
        Ntime, # time windows
        Nfreq, # frequency bins
        Ncues,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(FC3, self).__init__()
        self.encoder = Encoder(           
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        # self.attention = Attention(Ncues)
        # self.fc_freq = nn.Linear(Nfreq*Ncues, Nloc)
        # self.fc_time = nn.Linear(Ntime, 1)
        self.fc_time_freq = nn.Linear(Ntime*Nfreq*Ncues, 256)
        self.fc2 = nn.Linear(256, Nloc)
        self.fc3 = nn.Linear(Nloc, Nloc)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,0].permute(0,2,1))
            encList.append(enc)

        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)
        out = out.permute(1,2,3,0)

        out = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out.shape)

        out = self.fc_time_freq(out)
        out = self.bn(out)
        out = self.activation(out)
        # out = out.squeeze(-1)
        out = self.dropout(out)

        # out = torch.mean(attOut, -2)
        # out = out.squeeze(-1)
        if self.isDebug:
            print("FC freq shape: ",out.shape)
        
        # out = self.fcTime(out.permute(0,2,1))
        # out = out.squeeze(-1)
        # out = self.dropout(out)

        # out = torch.mean(out, -2)
        # out = out.squeeze(-2)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        if self.isDebug:
            print("FC time shape: ",out.shape)

        # out = self.softmaxLayer(out)
        return out

class DIYModel(nn.Module):
    def __init__(
        self,
        task,
        Ntime, # time windows
        Nfreq, # frequency bins
        Ncues,
        num_layers,
        numFC,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(DIYModel, self).__init__()
        self.encoder = Encoder(           
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )
        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        
        if numFC >= 3:
            self.FClayers = nn.ModuleList(
                [
                    nn.Linear(Ntime*Nfreq*Ncues, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, Nloc),
                    nn.ReLU()
                ]
            )
            self.FClayers.extend(
                [
                    nn.Linear(Nloc, Nloc)
                    for _ in range(numFC-2)
                ]
            )
        else:
            self.FClayers = nn.Sequential(
                nn.Linear(Ntime*Nfreq*Ncues, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(256, Nloc),
                nn.Tanh(),
                nn.Linear(Nloc, Nloc)
            )
        if task in ["elevRegression","azimRegression","allRegression"]:
            self.setRange1 = nn.Hardtanh()
            self.setRange2 = nn.Hardtanh()
        # self.sequentialFC = nn.Sequential(
        #     nn.Linear(Ntime*Nfreq*Ncues, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, Nloc)
        #     nn.ReLU()
        #     nn.Linear(Nloc, Nloc)
        # )

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,0].permute(0,2,1))
            encList.append(enc)
        
        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)
        
        out = out.permute(1,2,3,0)

        out = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out.shape)

        for layers in self.FClayers:
            out = layers(out)

        if self.task in ["elevRegression","azimRegression","allRegression"]:
            out = torch.stack(
                [
                    3/8*pi*self.setRange1(out[:,0])+pi/8,
                    pi*self.setRange2(out[:,1])+pi
                ], dim=1
            )
        
        # out = self.softmaxLayer(out)
        return out

class PytorchTransformer(nn.Module):
    def __init__(
        self,
        task,
        Ntime, # time windows
        Nfreq, # frequency bins
        Ncues,
        num_layers,
        numFC,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(PytorchTransformer, self).__init__()
        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        
        transformerLayers_torch = nn.TransformerEncoderLayer(
            d_model = Nfreq,
            nhead = 8,
            dim_feedforward = 4*Nfreq,
            dropout = dropout,
            activation = 'relu'
        )
        self.encoder = nn.TransformerEncoder(transformerLayers_torch, num_layers=num_layers)

        if numFC >= 3:
            self.FClayers = nn.ModuleList(
                [
                    nn.Linear(Ntime*Nfreq*Ncues, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, Nloc),
                    nn.ReLU()
                ]
            )
            self.FClayers.extend(
                [
                    nn.Linear(Nloc, Nloc)
                    for _ in range(numFC-2)
                ]
            )
        else:
            self.FClayers = nn.Sequential(
                nn.Linear(Ntime*Nfreq*Ncues, 256),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(256, Nloc),
                nn.Tanh(),
                nn.Linear(Nloc, Nloc)
            )
        if task in ["elevRegression","azimRegression","allRegression"]:
            self.setRange1 = nn.Hardtanh()
            self.setRange2 = nn.Hardtanh()
        self.isDebug = isDebug
        self.dropout = nn.Dropout(dropout)
        # self.softmaxLayer = nn.Softmax(dim = -1)
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,0].permute(2, 0, 1))
            encList.append(enc)

        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)
            
        out = torch.stack(encList)
        out = out.permute(2,1,3,0)

        out = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out.shape)

        for layers in self.FClayers:
            out = layers(out)

        if self.task in ["elevRegression","azimRegression","allRegression"]:
            out = torch.stack(
                [
                    3/8*pi*self.setRange1(out[:,0])+pi/8,
                    pi*self.setRange2(out[:,1])+pi
                ], dim=1
            )
        
        # out = self.softmaxLayer(out)
        return out

class DIY_parallel(nn.Module):
    def __init__(
        self,
        task,
        Ntime, # time windows
        Nfreq, # frequency bins
        Ncues,
        num_layers,
        numFC,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(DIY_parallel, self).__init__()
        self.encoder = Encoder(           
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )
        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        
        self.FClayers_elev = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.FClayers_azim = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        if task in ["elevRegression","azimRegression","allRegression"]:
            self.setRange_elev = nn.Hardtanh()
            self.setRange_azim = nn.Hardtanh()

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,0].permute(0,2,1))
            encList.append(enc)
        
        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)
        
        out = out.permute(1,2,3,0)

        out_elev = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out_elev.shape)

        for layers in self.FClayers_elev:
            out_elev = layers(out_elev)
        out_elev = 3/8*pi*self.setRange_azim(out_elev)+pi/8

        out_azim = torch.flatten(out, 1, -1)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        out_azim = pi*self.setRange_azim(out_azim)+pi

        if self.task in ["elevRegression","azimRegression","allRegression"]:
            out = torch.hstack((out_elev, out_azim))

        # out = self.softmaxLayer(out)
        return out

class DIY_multiSound(nn.Module):
    def __init__(
        self,
        task,
        Ntime, # time windows
        Nfreq, # frequency bins
        Ncues,
        Nsound,
        num_layers,
        numFC,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(DIY_multiSound, self).__init__()
        self.encoder = Encoder(           
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )
        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        
        self.FClayers_elev = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound)
        )
        self.FClayers_azim = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound)
        )

        if task in ["elevRegression","azimRegression","allRegression"]:
            self.setRange_elev = nn.Hardtanh()
            self.setRange_azim = nn.Hardtanh()

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,0].permute(0,2,1))
            encList.append(enc)
        
        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)
        
        out = out.permute(1,2,3,0)

        out_elev = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out_elev.shape)

        for layers in self.FClayers_elev:
            out_elev = layers(out_elev)
        out_elev = 3/8*pi*self.setRange_azim(out_elev)+pi/8

        out_azim = torch.flatten(out, 1, -1)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        out_azim = pi*self.setRange_azim(out_azim)+pi

        if self.task in ["elevRegression","azimRegression","allRegression"]:
            out = torch.stack((out_elev[:,0], out_azim[:,0],out_elev[:,1], out_azim[:,1]), dim=1)

        # out = self.softmaxLayer(out)
        return out



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    numEnc = 6
    numFC = 2
    Ntime = 44
    Nfreq = 512
    Ncues = 6
    batchSize = 32
    Nsound = 2
    # model = FC3(task, Ntime, Nfreq, Ncues, numLayers, 8, device, 4, 0, True).to(device)
    # model = DIYModel(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    # model = PytorchTransformer(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    # model = DIY_parallel(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    model = DIY_multiSound(task, Ntime, Nfreq, Ncues, Nsound, numEnc, numFC, 8, device, 4, 0, True).to(device)

    testInput = torch.rand(batchSize, Nfreq, Ntime, Ncues, dtype=torch.float32).to(device)
    print("testInput shape: ", testInput.shape)
    # print(testLabel)

    # print(testInput.permute(0,3,1,2).shape)
    # raise SystemExit("debug")
    testOutput = model(testInput)
    print("testOutput shape: ",testOutput.shape)
    print("testOutput: ",testOutput)
    # print(torch.max(testOutput, 1))

    # summary(model, (Nfreq, Ntime, Ncues))
    