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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from sklearn.utils import class_weight
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

from load_data import *
from utils import *
from utils_model import *
from models import *
from loss import *

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
    parser.add_argument('Nsound', type=int, help='Number of sound')
    parser.add_argument('whichDec', type=str, help='Which decoder structure')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--lrRate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--whichBest', default="bestValLoss", type=str, help='Best of acc or loss')
    parser.add_argument('--patience', default=10, type=int, help='Early stopping patience?')
    parser.add_argument('--Ncues', default=5, type=int, help='Number of cues')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--isContinue', default="True", type=str, help='Continue training?')
    parser.add_argument('--isSave', default="True", type=str, help='Save checkpoints?')
    parser.add_argument('--coordinates', default="spherical", type=str, help='Spherical or Cartesian')

    args = parser.parse_args()
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
    print("Early stopping patience: ", args.patience)
    print("Number of cues: ", args.Ncues)
    print("Number of sound: ", args.Nsound)
    print("Decoder structure: ", args.whichDec)

    """check input directories end up with /"""
    dir_var = {
        "data": args.dataDir,
        "model": args.modelDir
    }
    for idx in dir_var.keys():
        dir_var[idx] += "/"
    if not os.path.isdir(dir_var["model"]):
        os.mkdir(dir_var["model"])
    
    """create dicts holding the directory and flag variables"""
    flag_var = {
        "isDebug": args.isDebug,
        "isHPC": args.isHPC,
        "isContinue": args.isContinue,
        "isSave": args.isSave
    }
    for idx in flag_var.keys():
        flag_var[idx] = True if flag_var[idx][0].lower() == "t" else False

    """load dataset"""
    path = "./HRTF/IRC*"
    _, locLabel, _ = loadHRIR(path)

    train_dataset = CuesDataset(dir_var["data"] + "/train/",
                                args.task, args.Nsound, locLabel, coordinates=args.coordinates, isDebug=False)
    valid_dataset = CuesDataset(dir_var["data"] + "/valid/",
                                args.task, args.Nsound, locLabel, coordinates=args.coordinates, isDebug=False)
    print(f"Dataset length: {train_dataset.__len__()}, {valid_dataset.__len__()}")
    
    isPersistent = True if args.numWorker > 0 else False
    train_loader = MultiEpochsDataLoader(
        dataset=train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorker,
        persistent_workers=isPersistent
    )
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorker,
        persistent_workers=isPersistent
    )

    """definition"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nfreq = train_dataset.Nfreq
    Ntime = train_dataset.Ntime
    Ncues = train_dataset.Ncues
    Nsound = args.Nsound
    task = args.task

    """load model"""
    model = TransformerModel(
        task=task,
        Ntime=Ntime,
        Nfreq=Nfreq,
        Ncues=Ncues,
        Nsound=Nsound,
        whichEnc="diy",
        whichDec=args.whichDec,
        device=device,
        numEnc=args.numEnc,
        coordinates=args.coordinates,
        dropout=args.valDropout,
        forward_expansion=4,
        # numFC=args.numFC,
    )
    if flag_var['isHPC']:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    """load from checkpoint"""

    """define optimizer"""
    learning_rate = args.lrRate
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    """set learning rate scheduler"""
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    """set early stopping"""
    early_stop = EarlyStopping(args.patience)

    """set cost function"""
    cost_func = CostFunc(task=task, Nsound=Nsound, device=device, coordinates=args.coordinates)

    """tensorboard"""
    writer = SummaryWriter(f'runs/temp/tryingout_tensorboard')

    """training-validation iteration"""
    pre_epoch = 0
    for epoch in range(pre_epoch, pre_epoch + args.numEpoch):
        print("\nEpoch %d, lr = %f" % ((epoch + 1), getLR(optimizer)))
        train_correct = 0.0
        train_total = 0.0
        train_sum_loss = 0.0
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            num_batches = len(train_loader)
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            outputs = model(inputs)

            if flag_var["isDebug"]:
                print(
                    # "Input shape: ", inputs.shape, "\n",
                    # "label shape: ", labels.shape, "\n",
                    "labels: ", labels[:5], "\n",
                    # "Output shape: ", outputs.shape, "\n",
                    "Training Outputs: ", outputs[:5]
                )

            loss = cost_func(outputs, labels)
            train_sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        train_loss = train_sum_loss / (i + 1)
        print('Training Loss: %.04f | RMS angle (degree): %.04f '
                  % (train_loss, train_acc))
        
        val_correct = 0.0
        val_total = 0.0
        val_sum_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                outputs = model(inputs)
                
                if flag_var["isDebug"]:
                    print(
                        # "Input shape: ", inputs.shape, "\n",
                        # "label shape: ", labels.shape, "\n",
                        # "labels: ", labels[:5], "\n",
                        # "Output shape: ", outputs.shape, "\n",
                        "Validation Outputs: ", outputs[:5]
                    )

                val_loss = cost_func(outputs, labels)
                val_sum_loss += val_loss.item()
            val_loss = val_sum_loss / (i + 1)
            print('Validation Loss: %.04f | Validation Acc: %.04f '
                % (val_loss, val_acc))
            scheduler.step(val_loss)
        
        if args.isSave:
            saveCurves(
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                args.modelDir + "curve_epoch_" + str(epoch + 1) + ".pth.tar",
                args.task
            )
        writer.add_scalar('Training Loss', train_loss, global_step=epoch+1)
        writer.add_scalar('Validation Loss', val_loss, global_step=epoch+1)

        if early_stop.count == 0 and args.isSave:
            saveParam(
                epoch + 1,
                model,
                optimizer,
                scheduler,
                args.modelDir + "param_bestValLoss.pth.tar",
                task
            )
        early_stop(val_loss)