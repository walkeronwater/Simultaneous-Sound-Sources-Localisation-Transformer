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

from data_loader import *
from utils import *
from utils_train import *
from utils_test import *
from models import *
from loss import *


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
    parser.add_argument('--numFC', default=4, type=int, help='Number of FC layers')
    parser.add_argument('--lrRate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--whichBest', default="bestValLoss", type=str, help='Best of acc or loss')
    parser.add_argument('--patience', default=10, type=int, help='Early stopping patience?')
    parser.add_argument('--Ncues', default=4, type=int, help='Number of cues')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--isContinue', default="True", type=str, help='Continue training?')
    parser.add_argument('--isSave', default="True", type=str, help='Save checkpoints?')
    parser.add_argument('--coordinates', default="spherical", type=str, help='Spherical or Cartesian')
    parser.add_argument('--isLogging', default="False", type=str, help='Log down prediction in a csv file')
    parser.add_argument('--logName', default="test_log", type=str, help='Log down prediction in a csv file')

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
        raise SystemExit("Model not found.")
    
    """create dicts holding the directory and flag variables"""
    flag_var = {
        "isDebug": args.isDebug,
        "isHPC": args.isHPC,
        "isContinue": args.isContinue,
        "isSave": args.isSave,
        "isLogging": args.isLogging
    }
    for idx in flag_var.keys():
        flag_var[idx] = True if flag_var[idx][0].lower() == "t" else False
    
    """load dataset"""
    path = "./HRTF/IRC*"
    _, locLabel, _ = loadHRIR(path)

    test_dataset = CuesDataset(dir_var["data"],
                                args.task, args.Nsound, locLabel, coordinates=args.coordinates, isDebug=False)
    print(f"Dataset length: {test_dataset.__len__()}")
    
    isPersistent = True if args.numWorker > 0 else False
    test_loader = MultiEpochsDataLoader(
        dataset=test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorker,
        persistent_workers=isPersistent
    )

    """definition"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nfreq = test_dataset.Nfreq
    Ntime = test_dataset.Ntime
    Ncues = test_dataset.Ncues
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
        numFC=args.numFC,
    )
    elif args.whichModel.lower() == "crnn":
        model = CRNN(
            task=task,
            Ntime=Ntime,
            Nfreq=Nfreq,
            Ncues=Ncues,
            Nsound=Nsound,
            whichDec="src",
            num_conv_layers=4,
            num_recur_layers=2,
            num_FC_layers=args.numFC,
            dropout=args.valDropout,
            device=device,
            isDebug=False,
            coordinates="spherical"
        )
    else:
        raise SystemExit("Unsupported model.")

    if flag_var['isHPC']:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    """load from checkpoint"""
    checkpoint = torch.load(dir_var['model']+"param_bestValLoss.pth.tar")
    model.load_state_dict(checkpoint['model'], strict=True)

    """set cost function"""
    cost_func = CostFunc(task=task, Nsound=Nsound, device=device, coordinates=args.coordinates)
    
    error_src = TwoSourceError()
    csv_flag = False
    csv_name = args.logName + ".csv"
    """test iteration"""
    test_correct = 0.0
    test_total = 0.0
    test_sum_loss = 0.0
    test_loss = 0.0
    test_acc = 0.0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            outputs = model(inputs)
            
            if flag_var["isDebug"]:
                print(
                    "Input shape: ", inputs.shape, "\n",
                    "label shape: ", labels.shape, "\n",
                    "labels: ", labels[:5], "\n",
                    "Output shape: ", outputs.shape, "\n",
                    "Test Outputs: ", outputs[:5]
                )
            if flag_var["isLogging"]:
                if not csv_flag:
                    csv_flag = True
                    with open(csv_name, 'w') as csvFile:
                        for batch_idx in range(outputs.shape[0]):
                            for i in range(outputs.shape[1]):
                                csvFile.write(str(radian2Degree(outputs[batch_idx, i].item())))
                                csvFile.write(',')
                            for i in range(outputs.shape[1]):
                                csvFile.write(str(radian2Degree(labels[batch_idx, i].item())))
                                csvFile.write(',')
                            csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 0:2].unsqueeze(0), labels[batch_idx, 0:2].unsqueeze(0)).item())))
                            csvFile.write(',')
                            csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 2:4].unsqueeze(0), labels[batch_idx, 2:4].unsqueeze(0)).item())))
                            csvFile.write('\n')
                else:
                    with open(csv_name, 'a') as csvFile:
                        for batch_idx in range(outputs.shape[0]):
                            for i in range(outputs.shape[1]):
                                csvFile.write(str(radian2Degree(outputs[batch_idx, i].item())))
                                csvFile.write(',')
                            for i in range(outputs.shape[1]):
                                csvFile.write(str(radian2Degree(labels[batch_idx, i].item())))
                                csvFile.write(',')
                            csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 0:2].unsqueeze(0), labels[batch_idx, 0:2].unsqueeze(0)).item())))
                            csvFile.write(',')
                            csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 2:4].unsqueeze(0), labels[batch_idx, 2:4].unsqueeze(0)).item())))
                            csvFile.write('\n')

            test_loss = cost_func(outputs, labels)
            test_sum_loss += test_loss.item()

            loss_1 = radian2Degree(cost_func.calDoALoss(outputs[:, 0:2], labels[:, 0:2]))
            loss_2 = radian2Degree(cost_func.calDoALoss(outputs[:, 2:4], labels[:, 2:4]))

            """visualise predicted location vs ground truth"""
            error_src(outputs, labels, loss_1, loss_2)

    test_loss = test_sum_loss / (i + 1)
    test_acc = radian2Degree(test_loss)
    print('Test Loss: %.04f | RMS angle error in degree: %.04f '
        % (test_loss, test_acc))
    error_src.plotPrediction()
    error_src.plotError()