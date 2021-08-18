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
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
import torch.nn.functional as F
from torchsummary import summary
from graphviz import Source
from torchviz import make_dot

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
            Ntime,      
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
            enc = self.encoder(cues[:,:,:,i].permute(0,2,1))
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
            Ntime,     
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
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
            self.setRange1 = nn.Hardtanh(-pi/4, pi/2)
            self.setRange2 = nn.Hardtanh(0, pi*2)
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
            enc = self.encoder(cues[:,:,:,i].permute(0,2,1))
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
                    self.setRange1(out[:,0]),
                    self.setRange2(out[:,1])
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

        
        self.decoder_FC = DecoderFC(task, Ntime*Nfreq*Ncues, Nsound, dropout, isDebug)
        
        # if numFC >= 3:
        #     self.FClayers = nn.ModuleList(
        #         [
        #             nn.Linear(Ntime*Nfreq*Ncues, 256),
        #             nn.BatchNorm1d(256),
        #             nn.ReLU(),
        #             nn.Dropout(0.1),
        #             nn.Linear(256, Nloc),
        #             nn.ReLU()
        #         ]
        #     )
        #     self.FClayers.extend(
        #         [
        #             nn.Linear(Nloc, Nloc)
        #             for _ in range(numFC-2)
        #         ]
        #     )
        # else:
        #     self.FClayers = nn.Sequential(
        #         nn.Linear(Ntime*Nfreq*Ncues, 256),
        #         nn.BatchNorm1d(256),
        #         nn.Tanh(),
        #         nn.Dropout(0.1),
        #         nn.Linear(256, Nloc),
        #         nn.Tanh(),
        #         nn.Linear(Nloc, Nloc)
        #     )
        self.setRange1 = nn.Hardtanh(-pi / 4, pi / 2)
        self.setRange2 = nn.Hardtanh(0, pi * 2)
        self.isDebug = isDebug
        # self.softmaxLayer = nn.Softmax(dim = -1)
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,i].permute(2, 0, 1))
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

        out = torch.stack(
            [
                self.setRange1(out[:,0]),
                self.setRange2(out[:,1])
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
            Ntime,
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

        self.setRange_elev = nn.Hardtanh(-pi/4, pi/2)
        self.setRange_azim = nn.Hardtanh(0, pi*2)

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,i].permute(0,2,1))
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
        out_elev = self.setRange_elev(out_elev)

        out_azim = torch.flatten(out, 1, -1)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        out_azim = self.setRange_azim(out_azim)

        out = torch.hstack((out_elev, out_azim))

        # out = self.softmaxLayer(out)
        return out
'''
class DIY_parallel_multiSound(nn.Module):
    def __init__(
            self,
            task,
            Ntime,  # time windows
            Nfreq,  # frequency bins
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
        super(DIY_parallel, self).__init__()
        self.encoder = Encoder(
            Ntime,
            Nfreq,  # frequency bins
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
            nn.Linear(Ntime * Nfreq * Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.FClayers_azim = nn.Sequential(
            nn.Linear(Ntime * Nfreq * Ncues, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.setRange_elev = nn.Hardtanh(-pi / 4, pi / 2)
        self.setRange_azim = nn.Hardtanh(0, pi * 2)

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug

    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:, :, :, i].permute(0, 2, 1))
            encList.append(enc)

        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)

        out = out.permute(1, 2, 3, 0)

        out_elev = torch.flatten(out, 1, -1)
        if self.isDebug:
            print("Encoder output shape: ", out_elev.shape)

        for layers in self.FClayers_elev:
            out_elev = layers(out_elev)
        out_elev = self.setRange_elev(out_elev)

        out_azim = torch.flatten(out, 1, -1)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        out_azim = self.setRange_azim(out_azim)

        out = torch.hstack((out_elev, out_azim))

        # out = self.softmaxLayer(out)
        return out

'''
'''
# trained for submit_0108
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
        isDebug,
        batchSize
    ):
        super(DIY_multiSound, self).__init__()
        self.encoder = Encoder(
            Ntime,
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )

        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        

        self.FClayers_elev = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound)
        )
        self.FClayers_azim = nn.Sequential(
            nn.Linear(Ntime*Nfreq*Ncues, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, Nsound)
        )

        # self.test_layer_elev = nn.Linear(Ntime*Nfreq*Ncues, Nsound)
        # self.test_layer_azim = nn.Linear(Ntime*Nfreq*Ncues, Nsound)
        # self.setRange_elev = nn.Hardtanh(-pi/4, pi/2)
        # self.setRange_azim = nn.Hardtanh(0, pi*2)

        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,i].permute(0,2,1))
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
        # out_elev = self.test_layer_elev(out_elev)
        # out_elev = self.setRange_elev(out_elev)

        out_azim = torch.flatten(out, 1, -1)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        # out_azim = self.test_layer_azim(out_azim)
        # out_azim = self.setRange_azim(out_azim)

        # out = torch.hstack((out_elev, out_azim))
        out = torch.stack((out_elev[:,0], out_azim[:,0],out_elev[:,1], out_azim[:,1]), dim=1)

        # out = self.softmaxLayer(out)
        return out
'''

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
            Ntime, 
            Nfreq, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )

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

        self.task = task
        Nloc = predNeuron(task)
        print("Number of neurons in the final layer: ", Nloc)
        
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

        self.test_layer_elev = nn.Linear(Ntime*Nfreq*Ncues, Nsound)
        self.test_layer_azim = nn.Linear(Ntime*Nfreq*Ncues, Nsound)
        self.setRange_elev = nn.Hardtanh(-pi/4, pi/2)
        self.setRange_azim = nn.Hardtanh(0, pi*2)

        self.decoder_FC = DecoderFC(task, Ntime * Nfreq * Ncues, Nsound, dropout, isDebug)
        # self.softmaxLayer = nn.Softmax(dim = -1)
        self.isDebug = isDebug
    def forward(self, cues):
        # out = self.convLayers(cues.permute(0,3,2,1))
        # if self.isDebug:
        #     print("Shape after convLayers: ", out.shape)

        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,i].permute(0,2,1))
            encList.append(enc)
        
        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)

        out = torch.stack(encList)
        
        out = out.permute(1,2,3,0)

        return self.decoder_FC(out)

        # out_elev = torch.flatten(out, 1, -1)
        # if self.isDebug:
        #     print("Encoder output shape: ", out_elev.shape)
        #
        # for layers in self.FClayers_elev:
        #     out_elev = layers(out_elev)
        # # out_elev = self.test_layer_elev(out_elev)
        # out_elev = self.setRange_elev(out_elev)
        #
        # out_azim = torch.flatten(out, 1, -1)
        # for layers in self.FClayers_azim:
        #     out_azim = layers(out_azim)
        # # out_azim = self.test_layer_azim(out_azim)
        # out_azim = self.setRange_azim(out_azim)
        #
        # # out = torch.hstack((out_elev, out_azim))
        # out = torch.stack((out_elev[:,0], out_azim[:,0],out_elev[:,1], out_azim[:,1]), dim=1)
        #
        # # out = self.softmaxLayer(out)
        # return out

class Pytorch_transformer_multiSound(nn.Module):
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
        super(Pytorch_transformer_multiSound, self).__init__()
        
        transformerLayers_torch = nn.TransformerEncoderLayer(
            d_model = Nfreq,
            nhead = 8,
            dim_feedforward = 4*Nfreq,
            dropout = dropout,
            activation = 'relu'
        )
        self.encoder = nn.TransformerEncoder(transformerLayers_torch, num_layers=num_layers)

        self.decoder_FC = DecoderFC(task, Ntime*Nfreq*Ncues, Nsound, dropout, isDebug)

        self.isDebug = isDebug

    def forward(self, cues):
        encList = []
        for i in range(cues.shape[-1]):
            enc = self.encoder(cues[:,:,:,i].permute(2, 0, 1))
            encList.append(enc)

        if self.isDebug:
            print("Encoder for one cue shape: ", enc.shape)
            
        out = torch.stack(encList)
        out = out.permute(2,1,3,0)

        if self.isDebug:
            print("Encoder output shape: ", out.shape)

        return self.decoder_FC(out)


class DecoderFC(nn.Module):
    def __init__(
        self,
        task,
        flattenShape,
        Nsound,
        dropout,
        device,
        isDebug=False
    ):
        super(DecoderFC, self).__init__()

        if task.lower() in ["allregression", "elevregression", "azimregression"]:
            predLayer = nn.Linear(256, Nsound, bias=False)
        elif task.lower() in ["allclass"]:
            # [TODO] hardcoded Nloc = 187
            predLayer = nn.Linear(256, 187, bias=False)
        self.FClayers_elev = nn.Sequential(
            # nn.Linear(7872, 256),
            nn.Linear(flattenShape, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            predLayer
            # nn.Linear(256, Nsound, bias=False)
        )
        self.FClayers_azim = nn.Sequential(
            # nn.Linear(7872, 256),
            nn.Linear(flattenShape, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            predLayer
            # nn.Linear(256, Nsound, bias=False)
        )
        self.task = task
        self.setRange_elev = nn.Hardtanh(-pi/4, pi/2)
        self.setRange_azim = nn.Hardtanh(0, pi*2)

    def forward(self, x):
        out_elev = torch.flatten(x, 1, -1)
        out_azim = torch.flatten(x, 1, -1)

        for layers in self.FClayers_elev:
            out_elev = layers(out_elev)
        # out_elev = self.test_layer_elev(out_elev)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)
        # out_azim = self.test_layer_azim(out_azim)
        if self.task.lower() in ["allregression","elevregression","azimregression"]:
            out_elev = self.setRange_elev(out_elev)
            out_azim = self.setRange_azim(out_azim)
            out = torch.stack((out_elev[:, 0], out_azim[:, 0], out_elev[:, 1], out_azim[:, 1]), dim=1)
        else:
            out = torch.stack((out_elev, out_azim), dim=1)

        print("out shape: ", out.shape)

        return out