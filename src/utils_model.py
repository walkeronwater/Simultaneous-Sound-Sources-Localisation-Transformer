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

def predNeuron(task):
    if task == "elevClass":
        return 10
    elif task == "azimClass":
        return 24
    elif task == "allClass":
        return 187
    elif task == "elevRegression" or task == "azimRegression":
        return 1
    elif task == "allRegression":
        return 2



def saveParam(epoch, model, optimizer, scheduler, savePath, task):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'task': task
    }, savePath)

def saveCurves(epoch, tl, ta, vl, va, savePath, task):
    torch.save({
        'epoch': epoch,
        'train_loss': tl,
        'train_acc': ta,
        'valid_loss': vl,
        'valid_acc': va,
        'task': task
    }, savePath)

def loadCheckpoint(model, optimizer, scheduler, loadPath, task, phase, whichBest=None):
    if whichBest == "None":
        checkpoint = torch.load(loadPath+"param.pth.tar")
    else:
        checkpoint = torch.load(loadPath+"param"+"_"+whichBest+".pth.tar")

    if checkpoint['task'] == task:
        epoch = checkpoint['epoch']
        print("Model is retrieved at epoch ", epoch)
        # try:
        model.load_state_dict(checkpoint['model'], strict=False)
        # except:
        #     model.load_state_dict(checkpoint['state_dict'])

        trainHistory = glob(os.path.join(loadPath, "curve*"))

        history = {
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'valid_loss': []
        }
        for i in range(len(trainHistory)):
            checkpt = torch.load(
                loadPath+"curve_epoch_"+str(i+1)+".pth.tar"
            )
            for idx in history.keys():
                history[idx].append(checkpt[idx])

        val_loss_optim = history['valid_loss'][epoch-1]
        print("val_loss_optim: ", val_loss_optim)
        print("Corresponding validation accuracy: ",
            history['valid_acc'][epoch-1]
        )

        if phase == "train":
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print("scheduler not found")
            
            preTrainEpoch = len(trainHistory)
            print("Training will start from epoch", preTrainEpoch+1)
            return model, optimizer, scheduler, preTrainEpoch, val_loss_optim
        elif phase == "test":
            return model, val_loss_optim
    else:
        raise SystemExit("Task doesn't match")

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.stop = False
        self.count = 0
        self.val_loss_optim = float('inf')

    def __call__(self, val_loss):
        if self.val_loss_optim < val_loss:
            self.count += 1
        else:
            self.val_loss_optim = val_loss
            self.count = 0

        if self.count >= self.patience:
            self.stop = True

def getLR(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def setLR(newLR, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = newLR
    return optimizer