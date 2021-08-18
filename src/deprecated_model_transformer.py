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
from graphviz import Source
from torchviz import make_dot

from utils_train import *

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
        Ntime,
        Nfreq, # frequency bins
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout
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
        # fixed positional encoding
        # self.positional_encodings = torch.linspace(0, 1, Ntime)
        # self.positional_encodings = self.positional_encodings.repeat((batchSize, Nfreq, 1))
        # self.positional_encodings = self.positional_encodings.permute(0, 2, 1).to(device)
        # learnable postional embedding
        # self.position_embedding = nn.Embedding(max_length, Nfreq)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, Ntime, Nfreq = x.shape

        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        positional_encodings = torch.linspace(0, 1, Ntime)
        positional_encodings = positional_encodings.repeat((N, Nfreq, 1))
        positional_encodings = positional_encodings.permute(0, 2, 1).to(self.device)

        # print("positional encoding shape: ", self.positional_encodings.shape)
        out = x + positional_encodings
        out = self.dropout(out)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)

        return out

#[TODO]
'''
# Collection of encoders:
DIY transformer
Pytorch transformer

# Collection of decoders:
Classfication with 1 branch
Regression source-specific FC layers with Nsound branches
Regression separate elevation and azimuth specific FC layers with Nsound branches

'''

class Enc_DIY(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
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
        super(Enc_DIY, self).__init__()
        self.encoder = Encoder(
            Ntime,
            Nfreq,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )
    def forward(self, inputs):
        encList = []
        for i in range(inputs.shape[-1]):
            enc = self.encoder(inputs[:, :, :, i].permute(0, 2, 1))
            encList.append(enc)
        out = torch.stack(encList)
        out = out.permute(1, 2, 3, 0)
        return out

class Dec_1branch_cls(nn.Module):
    def __init__(
        self,
        enc_out_size,
        dropout
    ):
        super(Dec_1branch_cls, self).__init__()
        self.FClayers = nn.Sequential(
            nn.Linear(enc_out_size, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(256, 187)
        )

    def forward(self, enc_out):
        out_src = torch.flatten(enc_out, 1, -1)
        out_src = self.FClayers(out_src)
        return out_src

class Dec_2branch_src_reg(nn.Module):
    def __init__(
        self,
        enc_out_size,
        dropout
    ):
        super(Dec_2branch_src_reg, self).__init__()

        self.FClayers_src1 = nn.Sequential(
            nn.Linear(enc_out_size, 256),
            nn.BatchNorm1d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.FClayers_src2 = nn.Sequential(
            nn.Linear(enc_out_size, 256),
            nn.BatchNorm1d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, enc_out):
        out_src1 = torch.flatten(enc_out, 1, -1)
        out_src2 = torch.flatten(enc_out, 1, -1)

        for layers in self.FClayers_src1:
            out_src1 = layers(out_src1)
        for layers in self.FClayers_src2:
            out_src2 = layers(out_src2)

        out = torch.cat([out_src1, out_src2], dim=-1)
        return out

class Dec_2branch_ea_reg(nn.Module):
    def __init__(
        self,
        enc_out_size,
        dropout
    ):
        super(Dec_2branch_ea_reg, self).__init__()

        self.FClayers_elev = nn.Sequential(
            nn.Linear(enc_out_size, 256),
            nn.BatchNorm1d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.FClayers_azim = nn.Sequential(
            nn.Linear(enc_out_size, 256),
            nn.BatchNorm1d(256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.clamp_elev = nn.Hardtanh(-pi/4, pi/2)
        self.clamp_azim = nn.Hardtanh(0, pi*2)
    def forward(self, enc_out):
        out_elev = torch.flatten(enc_out, 1, -1)
        out_azim = torch.flatten(enc_out, 1, -1)

        for layers in self.FClayers_elev:
            out_elev = layers(out_elev)
        for layers in self.FClayers_azim:
            out_azim = layers(out_azim)

        out_elev = self.clamp_elev(out_elev)
        out_azim = self.clamp_azim(out_azim)

        out = torch.stack(
            (out_elev[:, 0], out_azim[:, 0], out_elev[:, 1], out_azim[:, 1])
            , dim=-1
        )
        return out

class TransformerModel(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        numEnc,
        numFC,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug,
        whichDec
    ):
        super(TransformerModel, self).__init__()

        self.enc = Enc_DIY(
            task,
            Ntime,
            Nfreq,
            Ncues,
            Nsound,
            numEnc,
            numFC,
            heads,
            device,
            forward_expansion,
            dropout,
            isDebug
        )

        if whichDec.lower() == "ea":
            self.dec = Dec_2branch_ea_reg(
                enc_out_size=Nfreq*Ntime*Ncues,
                dropout=dropout
            )
        elif whichDec.lower() == "src":
            self.dec = Dec_2branch_src_reg(
                enc_out_size=Nfreq * Ntime * Ncues,
                dropout=dropout
            )
        elif whichDec.lower() == "cls":
            self.dec = Dec_1branch_cls(
                enc_out_size=Nfreq * Ntime * Ncues,
                dropout=dropout
            )
        else:
            raise SystemError("Invalid decoder structure")

    def forward(self, inputs):
        enc_out = self.enc(inputs)
        out = self.dec(enc_out)
        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    numEnc = 6
    numFC = 2
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    batchSize = 32
    Nsound = 2
    # model = FC3(task, Ntime, Nfreq, Ncues, numLayers, 8, device, 4, 0, True).to(device)
    # model = DIYModel(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    # model = PytorchTransformer(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    # model = DIY_parallel(task, Ntime, Nfreq, Ncues, numEnc, numFC, 8, device, 4, 0, True).to(device)
    # model = DIY_multiSound(task, Ntime, Nfreq, Ncues, Nsound, numEnc, numFC, 8, device, 4, 0, True, batchSize=32).to(device)
    model = TransformerModel(
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        numEnc,
        numFC,
        heads=8,
        device=device,
        forward_expansion=4,
        dropout=0.1,
        isDebug=False,
        whichDec="cls"
    ).to(device)

    testInput = torch.rand(batchSize, Nfreq, Ntime, Ncues, dtype=torch.float32).to(device)
    print("testInput shape: ", testInput.shape)
    # print(testLabel)

    # print(testInput.permute(0,3,1,2).shape)
    # raise SystemExit("debug")
    testOutput = model(testInput)
    print("testOutput shape: ",testOutput.shape)
    print("testOutput: ",testOutput)
    # print(torch.max(testOutput, 1))

    # print(model)
    # make_dot(testOutput.mean(), params=dict(model.named_parameters())).render("transformer_multi_sound", format="png")

    # summary(model, (Nfreq, Ntime, Ncues))
    