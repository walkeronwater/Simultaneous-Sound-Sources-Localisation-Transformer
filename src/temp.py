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

from utils import *
from model_transformer import *

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filePath, isDebug=False):
        super(MyDataset, self).__init__()
        self.filePath = filePath
        self.annotation = pd.read_csv(filePath+"dataLabels.csv", header=None)
        self.isDebug = isDebug

    def __len__(self):
        return int(self.annotation.iloc[-1, 0] + 1)
    
    def __getitem__(self, pathIndex):
        data = torch.load(self.filePath+str(pathIndex)+".pt")
        labels = torch.tensor(int(self.annotation.iloc[pathIndex, 1])) # classification
        # labels = torch.tensor(locLabel[int(self.annotation.iloc[idx, 1]), 1], dtype=torch.float32) # regression
        # labels = torch.tensor(int(((locLabel[self.annotation.iloc[dataIndex, 1], 0]+45) % 150)/15)) # classify elevation only
        # labels = torch.tensor(int((locLabel[self.annotation.iloc[dataIndex, 1], 1] % 360)/15)) # classify azimuth only

        if self.isDebug:
            print("pathIndex: ", pathIndex)

        return data, labels

def splitDataset(batchSize, trainValidSplit: list, numWorker, dataset):
    Ntrain = round(trainValidSplit[0]*dataset.__len__())
    if Ntrain % batchSize == 1:
        Ntrain -=1
    Nvalid = round(trainValidSplit[1]*dataset.__len__())
    if Nvalid % batchSize == 1:
        Nvalid -=1
    # Ntest = dataset.__len__() - Ntrain - Nvalid
    # if Ntest % batchSize == 1:
    #     Ntest -=1
    print("Dataset separation: ", Ntrain, Nvalid)

    train, valid = torch.utils.data.random_split(dataset, [Ntrain, Nvalid], generator=torch.Generator().manual_seed(24))
    # train_loader = DataLoader(dataset=train, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)
    # valid_loader = DataLoader(dataset=valid, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)

    train_loader = MultiEpochsDataLoader(dataset=train, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)
    valid_loader = MultiEpochsDataLoader(dataset=valid, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)

    return train_loader, valid_loader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training hyperparamters')
    parser.add_argument('dataDir', type=str, help='Directory of saved cues')
    parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    print("Data directory", args.dataDir)
    print("Model directory", args.modelDir)
    print("Number of workers", args.numWorker)
    trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
    print("Train validation split", trainValidSplit)
    print("Number of encoder layers", args.numEnc)
    print("Number of FC layers", args.numFC)
    print("Dropout value", args.valDropout)
    print("Number of epochs", args.numEpoch)
    print("Batch size", args.batchSize)

    check_time = time.time()
    # dirName = './saved_cues/'
    dirName = args.dataDir
    assert (
        os.path.isdir(dirName)
    ), "Data directory doesn't exist."

    dataset = MyDataset(dirName)
    # train_loader, valid_loader = splitDataset(args.batchSize, trainValidSplit, args.numWorker, dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorker)

    print("Dataset instantialised - time elapse: ", round(time.time() - check_time, 5))
    check_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nsample = dataset.__len__()
    Nloc = 24
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    valDropout = args.valDropout
    num_layers = args.numEnc
    model = FC3(Nloc, Ntime, Nfreq, Ncues, num_layers, 8, device, 4, valDropout, False).to(device)

    print("Model instantialised - time elapse: ", round(time.time() - check_time, 5))
    check_time = time.time()

    num_epochs = args.numEpoch
    learning_rate = 1e-4
    early_epoch = 10
    early_epoch_count = 0
    val_acc_optim = 0.0

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    checkpointPath = args.modelDir + "/temp/"
    if not os.path.isdir(checkpointPath):
        os.mkdir(checkpointPath)

    
    print("Before entering training - time elapse: ", round(time.time() - check_time, 5))
    check_time = time.time()
    for epoch in range(num_epochs):
        print("\nEpoch %d, lr = %f" % ((epoch + 1), get_lr(optimizer)))
        
        train_correct = 0.0
        train_total = 0.0
        train_sum_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        start_time_enum = time.time()
        model.train()
        for i, data in enumerate(train_loader, 0):
            print("Pre loading time: ", round(time.time() - start_time_enum, 5))
            
            start_time_load = time.time()
            num_batches = len(train_loader)
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print("Input shape: ",inputs.shape)

            print("Loading one batch time: ", round(time.time() - start_time_load, 5))
            start_time_model = time.time()
            outputs = model(inputs)
            
            # print("Ouput shape: ", outputs.shape)
            # print("Label shape: ", labels.shape)
            loss = criterion(outputs, labels) # .unsqueeze_(1)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            train_sum_loss += loss.item()

            print("Model prediction time: ", round(time.time()-start_time_model, 5))

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.data).sum().item()
            start_time_enum = time.time()
        train_loss = train_sum_loss / (i+1)
        train_acc = round(100.0 * train_correct / train_total, 2)
        print('Training Loss: %.04f | Training Acc: %.4f%% '
            % (train_loss, train_acc))
        
'''
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
'''