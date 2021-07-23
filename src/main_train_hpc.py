import argparse
import re
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
from loss import *
from main_cues import CuesShape

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training hyperparamters')
    parser.add_argument('dataDir', type=str, help='Directory of saved cues')
    parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('whichModel', type=str, help='whichModel')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--lrRate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--whichBest', default="None", type=str, help='Best of acc or loss')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--ngpus', default=0, type=int, help='Number of GPUs')

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
    print("Model: ", args.whichModel)
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

    train_dataset = MyDataset(dirName+"/train/", args.task, args.isDebug)
    valid_dataset = MyDataset(dirName+"/valid/", args.task, args.isDebug)
    print("Dataset length: ", train_dataset.__len__())
    print("Dataset length: ", valid_dataset.__len__())

    isPersistent = True if args.numWorker > 0 else False
    train_loader = MultiEpochsDataLoader(
        dataset=train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorker, persistent_workers=isPersistent
    )
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorker, persistent_workers=isPersistent
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nfreq = CuesShape.Nfreq
    Ntime = CuesShape.Ntime
    Ncues = CuesShape.Ncues
    # model = FC3(args.task, Ntime, Nfreq, Ncues, args.numEnc, 8, device, 4, args.valDropout, args.isDebug).to(device)
    # model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug).to(device)
    if args.whichModel.lower() == "transformer":
        model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    elif args.whichModel.lower() == "cnn":
        model = CNNModel(task=args.task, dropout=args.valDropout, isDebug=False)
    # num_epochs = 30
    num_epochs = args.numEpoch
    pretrainEpoch = 0
    learning_rate = args.lrRate
    early_epoch = 10
    early_epoch_count = 0
    val_loss_optim = float('inf')
    val_acc_optim = 0.0

    num_warmup_steps = 5
    num_training_steps = num_epochs+1
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = AdamW(model.parameters())

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, 
    #     num_training_steps=num_training_steps
    # )
    
    lr_lambda = lambda epoch: learning_rate * np.minimum(
        (epoch + 1) ** -0.5, (epoch + 1) * (num_warmup_steps ** -1.5)
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    if not os.path.isdir(args.modelDir):
        os.mkdir(args.modelDir)
    else:
        try:
            model, optimizer, scheduler, pretrainEpoch, val_loss_optim = loadCheckpoint(
                model, optimizer, scheduler, args.modelDir, args.task, phase="train", whichBest=args.whichBest
            )
            if args.lrRate != 1e-4:
                optimizer = setLR(args.lrRate, optimizer)
            print("Found a pre-trained model in directory", args.modelDir)
        except:
            print("Not found any pre-trained model in directory", args.modelDir)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
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
            if args.task in ["elevRegression","azimRegression","allRegression"]:
                loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
            else:
                loss = nn.CrossEntropyLoss(outputs, labels)
            if args.isDebug:
                print("Loss", loss.shape)
            train_sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if not (args.task in ["elevRegression","azimRegression","allRegression"]):
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels.data).sum().item()
        train_loss = train_sum_loss / (i+1)
        if args.task in ["elevRegression","azimRegression","allRegression"]:
            train_acc = radian2Degree(train_loss)
            print('Training Loss: %.04f | RMS angle (degree): %.04f '
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
                if args.task in ["elevRegression","azimRegression","allRegression"]:
                    val_loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
                else:
                    val_loss = nn.CrossEntropyLoss(outputs, labels)
                val_sum_loss += val_loss.item()

                if not (args.task in ["elevRegression","azimRegression","allRegression"]):
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels.data).sum().item()
            val_loss = val_sum_loss / (i+1)
            if args.task in ["elevRegression","azimRegression","allRegression"]:
                val_acc = radian2Degree(val_loss)
                print('Validation Loss: %.04f | RMS angle (degree): %.04f '
                    % (val_loss, val_acc))
            else:
                val_acc = round(100.0 * val_correct / val_total, 2)
                print('Validation Loss: %.04f | Validation Acc: %.4f%% '
                    % (val_loss, val_acc))
                if val_acc > val_acc_optim or ((val_acc == val_acc_optim) and (val_loss <= val_loss_optim)):
                    val_acc_optim = val_acc
                    # for classfication, we also save the model with the best validation accuracy
                    saveParam(
                        epoch+1,
                        model,
                        optimizer,
                        scheduler,
                        args.modelDir + "param_bestValAcc.pth.tar",
                        args.task
                    )
            scheduler.step(val_loss)

        # save the model with the best validation loss
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
        if val_loss >= val_loss_optim:
            early_epoch_count += 1
        else:
            val_loss_optim = val_loss
            early_epoch_count = 0

            saveParam(
                epoch+1,
                model,
                optimizer,
                scheduler,
                args.modelDir + "param_bestValLoss.pth.tar",
                args.task
            )
            
        if (early_epoch_count >= early_epoch):
            break