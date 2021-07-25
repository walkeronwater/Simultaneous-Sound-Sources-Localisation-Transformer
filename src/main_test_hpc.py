import argparse
import re
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

from load_data import *
from utils import *
from model_transformer import *
from loss import *
from main_train import *
from main_cues import *


def regressionAcc(output, label, locLabel, device):
    correct = 0

    for i in range(output.shape[0]):
        minAngle = float('inf')
        tempOutput = torch.stack(locLabel.shape[0]*[output[i]], dim=0).to(device)
        tempLabel = torch.tensor(degree2Radian(locLabel)).to(device)
        loss = DoALoss(tempOutput, tempLabel)
        pred = torch.argmin(loss)
        if int(label[i, 0].item()) == int(pred.item()):
            correct += 1
        # print("label", int(label[i, 0].item()))
        # print("pred", pred.item())
    # raise SystemExit("debug")
    # print('Acc: ', correct/output.shape[0])
    return correct

class ConfusionEval:
    def __init__(self, numExample):
        self.numExample = numExample
        self.rms_UD = 0.0
        self.rms_LR = 0.0
        self.rms_FB = 0.0

    def LR_loss(self, output):
        return torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.cos(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )

    def FB_loss(self, output):
        return torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.sin(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )

    def up_down(self, pred, target):
        # print(target.shape)
        self.rms_UD += torch.sum(torch.square(pred[:,0] - target[:,0])).item()

    def left_right(self, pred, target):
        self.rms_LR += torch.sum(torch.square(self.LR_loss(pred) - self.LR_loss(target))).item()
        # print(self.rms_LR)
        # raise SystemExit("dbg")

        # pred_ = torch.empty(pred.shape[0])
        # target_ = torch.empty(pred.shape[0])
        # for i in range(pred.shape[0]):
        #     pred_[i] = self.convertLR(pred[i,1])
        #     target_[i] = self.convertLR(target[i,1])
        # self.rms_LR += torch.sum(torch.square(pred_ - target_)).item()
        pass

    def front_back(self, pred, target):
        self.rms_FB += torch.sum(torch.square(self.FB_loss(pred) - self.FB_loss(target))).item()

        # pred_ = torch.empty(pred.shape[0])
        # target_ = torch.empty(pred.shape[0])
        # for i in range(pred.shape[0]):
        #     pred_[i] = self.convertFB(pred[i,1])
        #     target_[i] = self.convertFB(target[i,1])
        # self.rms_FB += torch.sum(torch.square(pred_ - target_)).item()
    
    def report(self):
        print(
            "UD, LR, FB: ",
            radian2Degree(np.sqrt(self.rms_UD / self.numExample)),
            radian2Degree(np.sqrt(self.rms_LR / self.numExample)),
            radian2Degree(np.sqrt(self.rms_FB / self.numExample))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing hyperparamters')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('modelDir', type=str, help='Directory of saved model')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('whichModel', type=str, help='whichModel')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--samplePerSNR', default=10, type=int, help='Number of samples per SNR')
    parser.add_argument('--whichBest', default="None", type=str, help='Best of acc or loss')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    if args.audioDir[-1] != "/":
        args.audioDir += "/"
    if args.hrirDir[-1] != "/":
        args.hrirDir += "/"
    if args.modelDir[-1] != "/":
        args.modelDir += "/"
    print("Audio files directory: ", args.audioDir)
    print("HRIR files directory: ", args.hrirDir)
    print("Model directory: ", args.modelDir)
    print("Number of workers: ", args.numWorker)
    print("Task: ", args.task)
    print("Number of encoder layers: ", args.numEnc)
    print("Number of FC layers: ", args.numFC)
    print("Dropout value: ", args.valDropout)
    print("Number of epochs: ", args.numEpoch)
    print("Batch size: ", args.batchSize)
    print("Number of samples per SNR: ", args.samplePerSNR)

    if args.isDebug == "True":
        args.isDebug = True
    else:
        args.isDebug = False

    path = args.hrirDir + "/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    path = glob(os.path.join(args.audioDir+"/*"))
    Naudio = len(path)
    print("Number of audio files: ", Naudio)

    lenSliceInSec = CuesShape.lenSliceInSec
    Nfreq = CuesShape.Nfreq
    Ntime = CuesShape.Ntime
    Ncues = CuesShape.Ncues
    Nloc = CuesShape.Nloc
    Nsample = Nloc * args.samplePerSNR
    valSNRList = [-10,-5,0,5,10,15,20,25,100]

    # allocate tensors cues and labels in RAM
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    if args.task == "allRegression":
        labels_ = torch.zeros((Nsample,3), dtype=torch.float32)
    else:
        labels_ = torch.zeros((Nsample,))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.whichModel.lower() == "transformer":
        # model = FC3(args.task, Ntime, Nfreq, Ncues, args.numEnc, 8, device, 4, args.valDropout, args.isDebug).to(device)
        model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    elif args.whichModel.lower() == "cnn":
        model = CNNModel(task=args.task, dropout=0, isDebug=False)
    model = nn.DataParallel(model)

    # learning_rate = 1e-4
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    model, val_optim = loadCheckpoint(model=model, optimizer=None, scheduler=None, loadPath=args.modelDir, task=args.task, phase="test", whichBest=args.whichBest)
    model.to(device)
    for valSNR in valSNRList:
        fileCount = 0   # count the number of data samples
        for audioIndex in range(len(path)):
            if fileCount == Nsample:
                break
            # print("Audio index: ", audioIndex)
            audio, fs_audio = sf.read(path[audioIndex])
            # audio = librosa.resample(audio, fs_audio, fs_HRIR)

            audioSliceList = audioSliceGenerator(audio, fs_HRIR, lenSliceInSec)

            for sliceIndex in range(len(audioSliceList)):
                if fileCount == Nsample:
                    break
                audioSlice = audio[audioSliceList[sliceIndex]]

                for locIndex in range(Nloc):
                    if fileCount == Nsample:
                        break

                    hrirLeft_re = librosa.resample(hrirSet[locIndex, 0], fs_HRIR, fs_audio)
                    hrirRight_re = librosa.resample(hrirSet[locIndex, 1], fs_HRIR, fs_audio)
                    sigLeft = np.convolve(audioSlice, hrirLeft_re)
                    sigRight = np.convolve(audioSlice, hrirRight_re)

                    # print("Location index: ", locIndex)
                    # showSpectrogram(sigLeft, fs_HRIR)
                    # showSpectrogram(sigRight, fs_HRIR)
                
                    specLeft = calSpectrogram(sigLeft + noiseGenerator(sigLeft, valSNR))
                    specRight = calSpectrogram(sigRight + noiseGenerator(sigRight, valSNR))

                    ipdCues = calIPD(specLeft, specRight)
                    ildCues = calILD(specLeft, specRight)
                    r_l, theta_l  = cartesian2euler(specLeft)
                    r_r, theta_r  = cartesian2euler(specRight)

                    cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))

                    cues_[fileCount] = cues
                    if args.task == "allRegression":
                        labels_[fileCount][0] = locIndex
                        labels_[fileCount][1:3] = locIndex2Label(locLabel, locIndex, args.task)
                    else:
                        labels_[fileCount] = locIndex2Label(locLabel, locIndex, args.task)

                    fileCount += 1
                    # if fileCount % (Nloc*len(valSNRList)) == 0:
                    #     print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                    #           fileCount // (Nloc*len(valSNRList)))

        # create tensor dataset from data loaded in RAM
        if args.task == "allRegression":
            dataset = TensorDataset(cues_, labels_)
        else:
            dataset = TensorDataset(cues_, labels_.long())

        test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=args.numWorker)

        # test phase
        
        if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
            # confusion matrix
            confusion_matrix = torch.zeros(predNeuron(args.task), predNeuron(args.task))

        test_correct = 0.0
        test_total = 0.0
        test_sum_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        
        confusion = ConfusionEval(Nsample)
        # print("UD, LR, FB: ", confusion.rms_UD, confusion.rms_LR, confusion.rms_FB)
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                outputs = model(inputs)

                if args.task in ["elevRegression","azimRegression","allRegression"]:
                    loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels[:, 1:3]))))
                
                    confusion.up_down(outputs, labels[:, 1:3])
                    confusion.left_right(outputs, labels[:, 1:3])
                    confusion.front_back(outputs, labels[:, 1:3])
                else:
                    loss = nn.CrossEntropyLoss(outputs, labels)
                test_sum_loss += loss.item()

                if not (args.task in ["elevRegression","azimRegression","allRegression"]):
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.data).sum().item()

                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                # else:
                    # test_total += labels.shape[0]
                    # test_correct += regressionAcc(outputs, labels, locLabel, device)
        test_loss = test_sum_loss / (i+1)
        if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
            test_acc = radian2Degree(test_loss)
            print('For SNR: %d Test Loss: %.04f | RMS angle (degree): %.04f '
                % (valSNR, test_loss, test_acc))
        else:
            test_acc = round(100.0 * test_correct / test_total, 2)
            print('For SNR: %d Test Loss: %.04f | Test Acc: %.4f%% '
                % (valSNR, test_loss, test_acc))
                
        confusion.report()