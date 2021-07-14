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
    def __init__(self, embedSize, Nloc):
        super(Attention, self).__init__()
        self.embedSize = embedSize

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(embedSize, embedSize, bias=False)
        self.keys = nn.Linear(embedSize, embedSize, bias=False)
        self.queries = nn.Linear(embedSize, embedSize, bias=False)
        self.fc_out = nn.Linear(embedSize, Nloc)

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
        Nloc,
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
        # self.attention = Attention(Ncues, Nloc)
        self.fc_freq = nn.Linear(Nfreq*Ncues, Nloc)
        self.fc_time = nn.Linear(Ntime, 1)
        self.fc_time_freq = nn.Linear(Ntime*Nfreq*Ncues, Nloc*4)
        self.fc2 = nn.Linear(Nloc*4, Nloc*2)
        self.fc3 = nn.Linear(Nloc*2, Nloc)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(Nloc*4)
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numLayers = 6
    Nloc = 24
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    model = FC3(Nloc, Ntime, Nfreq, Ncues, numLayers, 8, device, 4, 0, True).to(device)

    testInput = torch.rand(2, Nfreq, Ntime, Ncues, dtype=torch.float32).to(device)
    # testInput = x[0].unsqueeze(0).to(device)
    # testLabel = x[1].to(device)
    print("testInput shape: ", testInput.shape)
    # print(testLabel)
    testOutput = model(testInput)
    # print(torch.max(testOutput, 1))

    # summary(model, (Nfreq, Ntime, Ncues))
    