import argparse
import time
import timeit
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
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

# from load_data import *
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
    print(dataset)
    # train_loader, valid_loader = splitDataset(args.batchSize, trainValidSplit, args.numWorker, dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorker)

    print("Dataset instantialised - time elapse: ", round(time.time() - check_time, 5))

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

    
    print("Before entering training - time elapsed: ", round(time.time() - check_time, 5))
    for epoch in range(num_epochs):
        print("\nEpoch %d, lr = %f" % ((epoch + 1), get_lr(optimizer)))
        num_batches = len(train_loader)
        
        train_correct = 0.0
        train_total = 0.0
        train_sum_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        print("Before entering the first batch - time elapse: ", round(time.time() - check_time, 5))
        check_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader, 0):
            if (i+1)%250 == 0:
                print(" %d batches, time elapsed: %f" % (i+1, round(time.time() - check_time, 5)))
                check_time = time.time()
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