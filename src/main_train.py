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

from load_data import *
from utils import *
from model_transformer import *
from loss import DoALoss

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

def loadCheckpoint(model, optimizer, scheduler, loadPath, task, phase):
    checkpoint = torch.load(loadPath+"param.pth.tar")

    if checkpoint['task'] == task:
        epoch = checkpoint['epoch']
        print("Model is retrieved at epoch ", epoch)
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict(checkpoint['state_dict'])

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

        val_optim = history['valid_loss'][epoch-1]
        print("val_optim: ", val_optim)
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
            return model, optimizer, scheduler, preTrainEpoch, val_optim
        elif phase == "test":
            return model, val_optim
    else:
        raise SystemExit("Task doesn't match")

def getLR(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training hyperparamters')
    parser.add_argument('dataDir', type=str, help='Directory of saved cues')
    parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--lrRate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    if args.dataDir[-1] != "/":
        args.dataDir += "/"
    if args.modelDir[-1] != "/":
        args.modelDir += "/"
    
    print("Data directory: ", args.dataDir)
    print("Model directory: ", args.modelDir)
    print("Number of workers: ", args.numWorker)
    print("Task: ", args.task)
    trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
    print("Train validation split: ", trainValidSplit)
    print("Number of encoder layers: ", args.numEnc)
    print("Number of FC layers: ", args.numFC)
    print("Learning rate: ", args.lrRate)
    print("Dropout value: ", args.valDropout)
    print("Number of epochs: ", args.numEpoch)
    print("Batch size: ", args.batchSize)
    if args.isDebug == "True":
        args.isDebug = True
    else:
        args.isDebug = False

    # dirName = './saved_cues/'
    dirName = args.dataDir
    assert (
        os.path.isdir(dirName)
    ), "Data directory doesn't exist."

    dataset = MyDataset(dirName, args.task, args.isDebug)
    print("Dataset length: ", dataset.__len__())

    train_loader, valid_loader = splitDataset(args.batchSize, trainValidSplit, args.numWorker, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nsample = dataset.__len__()
    Ntime = 44
    Nfreq = 512
    Ncues = 5
    # model = FC3(args.task, Ntime, Nfreq, Ncues, args.numEnc, 8, device, 4, args.valDropout, args.isDebug).to(device)
    model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug).to(device)

    # num_epochs = 30
    num_epochs = args.numEpoch
    pretrainEpoch = 0
    learning_rate = args.lrRate
    early_epoch = 20
    early_epoch_count = 0
    val_optim = float('inf')

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

    if not os.path.isdir(args.modelDir):
        os.mkdir(args.modelDir)
    else:
        try:
            model, optimizer, scheduler, pretrainEpoch, val_optim = loadCheckpoint(model, optimizer, scheduler, args.modelDir, args.task, "train")
            print("Found a pre-trained model in directory", args.modelDir)
        except:
            print("Not found any pre-trained model in directory", args.modelDir)

    for epoch in range(pretrainEpoch, pretrainEpoch + num_epochs):
        print("\nEpoch %d, lr = %f" % ((epoch+1), getLR(optimizer)))
        
        train_correct = 0.0
        train_total = 0.0
        train_sum_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            num_batches = len(train_loader)
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print(labels)
            # print("Input shape: ",inputs.shape)
            outputs = model(inputs)
            
            # print("Ouput shape: ", outputs.shape)
            # print("Label shape: ", labels.shape)
            if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
                loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
            else:
                loss = criterion(outputs, labels)
            if args.isDebug:
                print("Loss", loss.shape)
            train_sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if not (args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression"):
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels.data).sum().item()
        train_loss = train_sum_loss / (i+1)
        if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
            train_acc = train_loss
            print('Training Loss: %.04f | Training Acc: %.04f '
                % (train_loss, train_acc))
        else:
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
            for i, (inputs, labels) in enumerate(valid_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                
                outputs = model(inputs)
                if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
                    val_loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
                else:
                    val_loss = criterion(outputs, labels) # .unsqueeze_(1)
                val_sum_loss += val_loss.item()

                if not (args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression"):
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels.data).sum().item()
            val_loss = val_sum_loss / (i+1)
            if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
                val_acc = val_loss
                print('Val_Loss: %.04f | Val_Acc: %.04f '
                    % (val_loss, val_acc))
            else:
                val_acc = round(100.0 * val_correct / val_total, 2)
                print('Val_Loss: %.04f | Val_Acc: %.4f%% '
                    % (val_loss, val_acc))
            scheduler.step(val_loss)

        saveCurves(
            epoch+1, 
            train_loss, 
            train_acc, 
            val_loss, 
            val_acc,
            args.modelDir + "curve_epoch_" + str(epoch+1) + ".pth.tar",
            args.task
        )

        # early stopping
        if (val_loss >= val_optim):
            early_epoch_count += 1
        else:
            val_optim = val_loss
            early_epoch_count = 0

            saveParam(
                epoch+1,
                model,
                optimizer,
                scheduler,
                args.modelDir + "param.pth.tar",
                args.task
            )
            
        if (early_epoch_count >= early_epoch):
            break