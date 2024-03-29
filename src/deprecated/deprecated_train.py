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
from model_transformer import *
from model_CNN import *
from model_RNN import *
from loss import *
# from main_cues import CuesShape


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

    args = parser.parse_args()
    if args.dataDir[-1] != "/":
        args.dataDir += "/"
    if args.modelDir[-1] != "/":
        args.modelDir += "/"
    if args.isDebug == "True":
        args.isDebug = True
    else:
        args.isDebug = False
    if args.isHPC == "True":
        args.isHPC = True
    else:
        args.isHPC = False
    if args.isContinue == "True":
        args.isContinue = True
    else:
        args.isContinue = False
    if args.isSave == "True":
        args.isSave = True
    else:
        args.isSave = False

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

    path = "./HRTF/IRC*"
    _, locLabel, _ = loadHRIR(path)
    # dirName = './saved_cues/'
    dirName = args.dataDir
    assert (
        os.path.isdir(dirName)
    ), "Data directory doesn't exist."

    train_dataset = MyDataset(dirName + "/train/", args.task, args.Nsound, locLabel, args.isDebug)
    valid_dataset = MyDataset(dirName + "/valid/", args.task, args.Nsound, locLabel, args.isDebug)
    print("Dataset length: ", train_dataset.__len__())
    print("Dataset length: ", valid_dataset.__len__())

    isPersistent = True if args.numWorker > 0 else False
    train_loader = MultiEpochsDataLoader(
        dataset=train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.numWorker,
        persistent_workers=isPersistent
    )
    valid_loader = MultiEpochsDataLoader(
        dataset=valid_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.numWorker,
        persistent_workers=isPersistent
    )

    # for i, (inputs, labels) in enumerate(train_loader):
    #     print(labels)
    #     print(labels.shape)
    #     raise SystemExit("dbg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuesShape = CuesShape(Ncues=args.Ncues)
    Nfreq = cuesShape.Nfreq
    Ntime = cuesShape.Ntime
    Ncues = cuesShape.Ncues
    Nsound = args.Nsound

    num_epochs = args.numEpoch
    pretrainEpoch = 0
    learning_rate = args.lrRate
    early_epoch = 10
    early_epoch_count = 0
    val_loss_optim = float('inf')
    val_acc_optim = 0.0

    if args.whichModel.lower() == "transformer":
        model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout,
                         args.isDebug)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif args.whichModel.lower() == "paralleltransformer":
        if args.Nsound == 1:
            model = DIY_parallel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout,
                                 args.isDebug)
        else:
            model = DIY_multiSound(args.task, Ntime, Nfreq, Ncues, Nsound, args.numEnc, args.numFC, 8, device, 4,
                                   args.valDropout, args.isDebug)
        # model.apply(weight_init)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif args.whichModel.lower() == "cnn":
        if args.Nsound == 1:
            model = CNNModel(task=args.task, Ncues=Ncues, dropout=args.valDropout, device=device, isDebug=args.isDebug)
        else:
            model = CNN_multiSound(
                task=args.task,
                Ntime=Ntime,
                Nfreq=Nfreq,
                Ncues=Ncues,
                Nsound=Nsound,
                dropout=args.valDropout,
                device=device,
                isDebug=args.isDebug
            )
        model.apply(weight_init)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    elif args.whichModel.lower() == "pytorchtransformer":
        if args.Nsound == 1:
            model = PytorchTransformer(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4,
                                       args.valDropout, args.isDebug)
        else:
            model = Pytorch_transformer_multiSound(args.task, Ntime, Nfreq, Ncues, Nsound, args.numEnc, args.numFC, 8,
                                                   device, 4, args.valDropout, args.isDebug)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # elif args.whichModel.lower() == "multisound":
    #     model = DIY_multiSound(args.task, Ntime, Nfreq, Ncues, Nsound, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    #     # model.apply(weight_init)
    #     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # elif args.whichModel.lower() == "multisoundcnn":

    #     model.apply(weight_init)
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    else:
        raise SystemExit("No model selected")
    # raise SystemExit("debug")

    if args.isHPC:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)


    # Define training hyperparmeters:
    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Load checkpoint to resume training
    if not os.path.isdir(args.modelDir):
        os.mkdir(args.modelDir)
    if args.isContinue:
        try:
            model, optimizer, scheduler, pretrainEpoch, val_loss_optim = loadCheckpoint(
                model, optimizer, scheduler, args.modelDir, args.task, phase="train", whichBest=args.whichBest
            )

            # if args.lrRate != 1e-4:
            #     optimizer = setLR(args.lrRate, optimizer)
            print("Found a pre-trained model in directory", args.modelDir)
        except:
            print("Not found any pre-trained model in directory", args.modelDir)

    # set up early stopping
    early_stop = EarlyStopping(args.patience)
    # use tensorboard
    writer = SummaryWriter(f'runs/temp/tryingout_tensorboard')
    step = 1
    for epoch in range(pretrainEpoch, pretrainEpoch + num_epochs):
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
            # print(labels)

            outputs = model(inputs)
            # print(
            #     "Input shape: ", inputs.shape, "\n",
            #     "label shape: ", labels.shape, "\n",
            #     "labels: ", labels[:5], "\n",
            #     "Output shape: ", outputs.shape, "\n",
            #     "Outputs: ", outputs[:5]
            # )
            # print("Ouput: ", outputs)
            # print("Label: ", labels)
            # raise SystemExit("debug")
            if Nsound == 1:
                if args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]:
                    loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
                else:
                    loss = nn.CrossEntropyLoss(outputs, labels)
                if args.isDebug:
                    print("Loss", loss.shape)
            else:
                if args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]:
                    loss = torch.sqrt(torch.mean(torch.square(cost_multiSound(outputs, labels))))
                    # loss = torch.sqrt(torch.mean(torch.square(cost_manhattan(outputs, labels))))
                else:
                    loss = nn.CrossEntropyLoss(outputs, labels)
            train_sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # for name, p in model.named_parameters():
            #     print(name, p.grad.norm().item())
            # raise SystemExit("debug")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if not (args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]):
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels.data).sum().item()
        train_loss = train_sum_loss / (i + 1)
        if args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]:
            train_acc = radian2Degree(train_loss)
            print('Training Loss: %.04f | RMS angle (degree): %.04f '
                  % (train_loss, train_acc))
        else:
            train_acc = round(100.0 * train_correct / train_total, 2)
            print('Training Loss: %.04f | Training Acc: %.4f%% '
                  % (train_loss, train_acc))

        # print("Training Ouput: ", outputs[0:10])
        # print("Training Label: ", labels[0:10])

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
                # print("Ouput: ", outputs)
                # print("Label: ", labels)
                # raise SystemExit("debug")
                if Nsound == 1:
                    if args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]:
                        val_loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
                    else:
                        val_loss = nn.CrossEntropyLoss(outputs, labels)
                else:
                    val_loss = torch.sqrt(torch.mean(torch.square(cost_multiSound(outputs, labels))))
                    # val_loss = torch.sqrt(torch.mean(torch.square(cost_manhattan(outputs, labels))))
                val_sum_loss += val_loss.item()

                if not (args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]):
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels.data).sum().item()
            val_loss = val_sum_loss / (i + 1)
            if args.task in ["elevRegression", "azimRegression", "allRegression", "multisound"]:
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
                    if args.isSave:
                        saveParam(
                            epoch + 1,
                            model,
                            optimizer,
                            scheduler,
                            args.modelDir + "param_bestValAcc.pth.tar",
                            args.task
                        )
            # scheduler.step(val_loss)
            scheduler.step()

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

        # update tensorboard
        writer.add_scalar('Training Loss', train_loss, global_step=step)
        writer.add_scalar('Training RMS angle (degree)', train_acc, global_step=step)
        writer.add_scalar('Validation Loss', val_loss, global_step=step)
        writer.add_scalar('Validation RMS angle (degree)', val_acc, global_step=step)
        step += 1

        early_stop(val_loss)
        if early_stop.stop:
            break

        if (early_stop.count == 0 or epoch == 0) and args.isSave:
            saveParam(
                epoch + 1,
                model,
                optimizer,
                scheduler,
                args.modelDir + "param_bestValLoss.pth.tar",
                args.task
            )

    '''
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
    '''
