import argparse
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

from utils import *
from model_transformer import *


# create dataset from disk
isDisk = True
if isDisk:
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, filePath, isDebug=False):
            super(MyDataset, self).__init__()
            self.filePath = filePath
            self.ptFilePath = glob(os.path.join(self.filePath, '*.pt'))
            self.csvFilePath = glob(os.path.join(self.filePath, '*.csv'))
            self.annotation = pd.read_csv(self.csvFilePath[0], header=None)
            
            self.isDebug = isDebug
            # self.data = None
            # self.filePath = "/content/data/music_loc24_SNR10.h5"
            # with h5py.File(self.filePath, 'r') as hf:
            #     self.lenDataset = len(hf["data"])

        # def __getindex__(self, idx):
        #     return load_file(self.data_files[idx])

        def __len__(self):
            return len(self.ptFilePath)
        
        def __getitem__(self, pathIndex):
            # if self.data is None:
                # self.data = h5py.File(self.filePath, 'r')["data"]
                # self.labels = h5py.File(self.filePath, 'r')["labels"]
                # self.data, self.labels = np.array(self.data[idx]), np.array(self.labels[idx])
                # self.data = torch.from_numpy(data.astype(np.float32))
                # self.labels = torch.from_numpy(labels.astype(np.long))

                # print(self.data[idx].shape)
                # print(self.labels[idx])
            data = torch.load(self.ptFilePath[pathIndex])
            dataIndex = int(os.path.basename(self.ptFilePath[pathIndex])[0:-3])
            labels = torch.tensor(int(self.annotation.iloc[dataIndex, 1])) # classification
            # labels = torch.tensor(locLabel[int(self.annotation.iloc[idx, 1]), 1], dtype=torch.float32) # regression
            # labels = torch.tensor(int(((locLabel[self.annotation.iloc[dataIndex, 1], 0]+45) % 150)/15)) # classify elevation only
            # labels = torch.tensor(int((locLabel[self.annotation.iloc[dataIndex, 1], 1] % 360)/15)) # classify azimuth only

            if self.isDebug:
                print("pathIndex: ", pathIndex)
                print("Data path: ", os.path.basename(self.ptFilePath[pathIndex]))
                print("dataIndex: ", dataIndex)

            return data, labels

dirName = './saved_cues/'
if os.path.isdir(dirName):
    dataset = MyDataset(dirName)
    
    batch_size = 32
    Ntrain = round(0.6*dataset.__len__())
    if Ntrain % batch_size == 1:
        Ntrain -=1
    Nvalid = round(0.2*dataset.__len__())
    if Nvalid % batch_size == 1:
        Nvalid -=1
    Ntest = dataset.__len__() - Ntrain - Nvalid
    if Ntest % batch_size == 1:
        Ntest -=1
    print("Dataset separation: ",Ntrain, Nvalid, Ntest)

    train, valid, test = torch.utils.data.random_split(dataset, [Ntrain, Nvalid, Ntest], generator=torch.Generator().manual_seed(24))
    train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=32, shuffle=True, num_workers=0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


valDropoutList = []
num_layersList = [6]

rootDir = "./model/"
if not os.path.isdir(rootDir):
    os.mkdir(rootDir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for num_layers in num_layersList:
    Nsample = dataset.__len__()
    Nloc = 24
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    valDropout = 0.3
    model = FC3(Nloc, Ntime, Nfreq, Ncues, num_layers, 8, device, 4, valDropout, False).to(device)
    model.isDebug = False
    dataset.isDebug = False

    num_epochs = 30
    learning_rate = 1e-4
    early_epoch = 10
    early_epoch_count = 0
    val_acc_optim = 0.0

    num_warmup_steps = 2
    num_training_steps = num_epochs+1
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = AdamW(model.parameters())

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, 
    #     num_training_steps=num_training_steps
    # )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    print("Data volume: ", Nsample)
    print("Nloc: ", Nloc)
    # print("SNR: ", valSNRList)
    print("Number of layers: ", num_layers)
    print("Dropout: ", valDropout)
    expName = "vol_"+str(Nsample)+"_loc_"+str(Nloc)+"_layer_"+str(num_layers)+"/"
    checkpointPath = rootDir+expName
    if not os.path.isdir(checkpointPath):
        os.mkdir(checkpointPath)

    for epoch in range(num_epochs):
        print("\nEpoch %d, lr = %f" % ((epoch + 1), get_lr(optimizer)))
        
        train_correct = 0.0
        train_total = 0.0
        train_sum_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            num_batches = len(train_loader)
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print("Input shape: ",inputs.shape)

            outputs = model(inputs)

            # print("Ouput shape: ", outputs.shape)
            # print("Label shape: ", labels.shape)
            loss = criterion(outputs, labels) # .unsqueeze_(1)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_sum_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.data).sum().item()
        train_loss = train_sum_loss / (i+1)
        train_acc = round(100.0 * train_correct / train_total, 2)
        print('Training Loss: %.04f | Training Acc: %.4f%% '
            % (train_loss, train_acc))
        
        val_correct = 0.0
        val_total = 0.0
        val_sum_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # validation phase
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                
                outputs = model(inputs)
                val_loss = criterion(outputs, labels) # .unsqueeze_(1)
                val_sum_loss += val_loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels.data).sum().item()
            val_loss = val_sum_loss / (i+1)
            val_acc = round(100.0 * val_correct / val_total, 2)
            scheduler.step(val_loss)

        print('Val_Loss: %.04f | Val_Acc: %.4f%% '
            % (val_loss, val_acc))
        
        checkpoint_curve = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": val_loss,
            "valid_acc": val_acc
        }
        
        torch.save(
            checkpoint_curve,
            checkpointPath + "curve_epoch_"+str(epoch+1)+".pth.tar"
        )

        

        # early stopping
        if (val_acc <= val_acc_optim):
            early_epoch_count += 1
        else:
            val_acc_optim = val_acc
            early_epoch_count = 0

            checkpoint_param = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            torch.save(
                checkpoint_param,
                checkpointPath + "param.pth.tar"
            )
        if (early_epoch_count >= early_epoch):
            break
