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
    def __init__(self, numExample, savePath, expName, isSave=True):
        self.numExample = numExample
        self.rms_UD = 0.0
        self.rms_LR = 0.0
        self.rms_FB = 0.0
        self.pred_elev = []
        self.target_elev = []
        self.pred_azim = []
        self.target_azim = []
        self.pred_UD = []
        self.pred_LR = []
        self.pred_FB = []
        self.target_UD = []
        self.target_LR = []
        self.target_FB = []
        self.savePath = savePath+"/" if savePath[-1] != "/" else savePath
        self.expName = expName
        self.isSave = isSave

    def evaluate(self, pred, target):
        self.rms_UD += torch.sum(torch.square(pred[:,0] - target[:,0])).item()
        self.rms_LR += torch.sum(torch.square(self.LR_loss(pred) - self.LR_loss(target))).item()
        self.rms_FB += torch.sum(torch.square(self.FB_loss(pred) - self.FB_loss(target))).item()
        self.pred_elev.extend(radian2Degree(pred[:,0].squeeze(0)).cpu())
        self.target_elev.extend(radian2Degree(target[:,0].squeeze(0)).cpu())
        self.pred_azim.extend(radian2Degree(pred[:,1].squeeze(0)).cpu())
        self.target_azim.extend(radian2Degree(target[:,1].squeeze(0)).cpu())
        self.pred_UD.extend(radian2Degree(pred[:,0]).cpu())
        self.pred_LR.extend(radian2Degree(self.LR_loss(pred)).cpu())
        self.pred_FB.extend(radian2Degree(self.FB_loss(pred)).cpu())
        self.target_UD.extend(radian2Degree(target[:,0]).cpu())
        self.target_LR.extend(radian2Degree(self.LR_loss(target)).cpu())
        self.target_FB.extend(radian2Degree(self.FB_loss(target)).cpu())

    def LR_loss(self, output):
        angle_diff = torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.cos(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )
        for i in range(angle_diff.shape[0]):
            if pi < output[i, 1] < pi*2:
                angle_diff[i] = -angle_diff[i]
                # print("LR: ",radian2Degree(angle_diff[i]))
        return angle_diff

    def FB_loss(self, output):
        angle_diff = torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.sin(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )
        for i in range(angle_diff.shape[0]):
            if pi/2 <= output[i, 1] <= pi*3/2:
                angle_diff[i] = -angle_diff[i]
                # print("FB: ",radian2Degree(angle_diff[i]))
        return angle_diff

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
        UD_confusion = radian2Degree(np.sqrt(self.rms_UD / self.numExample))
        LR_confusion = radian2Degree(np.sqrt(self.rms_LR / self.numExample))
        FB_confusion = radian2Degree(np.sqrt(self.rms_FB / self.numExample))

        print(
            "UD, LR, FB: ",
            UD_confusion,
            LR_confusion,
            FB_confusion
        )

        x = np.linspace(-45, 90, 100)
        y = x
        plt.figure()
        plt.scatter(self.pred_elev, self.target_elev)
        plt.plot(x, y,'-r')
        plt.xticks(range(-45,91,15))
        plt.yticks(range(-45,91,15))
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title("Elevation "+self.expName)
        if self.isSave:
            plt.savefig(self.savePath+self.expName+"_elev.png")
        plt.close()

        x = np.linspace(0, 345, 100)
        y = x
        plt.figure()
        plt.scatter(self.pred_azim, self.target_azim)
        plt.plot(x, y,'-r')
        plt.xticks(range(0,360,30))
        plt.yticks(range(0,360,30))
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title("Azimuth "+self.expName)
        if self.isSave:
            plt.savefig(self.savePath+self.expName+"_azim.png")
        plt.close()

        x = np.linspace(-90, 90, 100)
        y = x
        plt.figure()
        plt.scatter(self.pred_LR, self.target_LR)
        plt.plot(x, y,'-r')
        plt.xticks(range(-90,91,15))
        plt.yticks(range(-90,91,15))
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title("LR confusion "+self.expName)
        if self.isSave:
            plt.savefig(self.savePath+self.expName+"_lr.png")
        plt.close()

        x = np.linspace(-90, 90, 100)
        y = x
        plt.figure()
        plt.scatter(self.pred_FB, self.target_FB)
        plt.plot(x, y,'-r')
        plt.xticks(range(-90,91,15))
        plt.yticks(range(-90,91,15))
        plt.xlabel("Ground truth")
        plt.ylabel("Prediction")
        plt.title("FB confusion "+self.expName)
        if self.isSave:
            plt.savefig(self.savePath+self.expName+"_fb.png")
        plt.close()

        return (UD_confusion, LR_confusion, FB_confusion)

def plotConfusion(snrList, UDList, LRList, FBList, savePath, isSave=True):
    if isSave:
        plt.figure()
        plt.plot(snrList, UDList)
        plt.plot(snrList, LRList)
        plt.plot(snrList, FBList)
        plt.xlabel("SNR")
        plt.ylabel("RMS error of angle (degree)")
        plt.title("Confusion vs SNR")
        plt.legend(["UD", "LR", "FB"])
        plt.grid()
        plt.savefig(savePath+"confusion.png")
        plt.close()

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
    parser.add_argument('--valSNRList', default="-5,0,5,10,15,20,25,30,35", type=str, help='Range of SNR')
    parser.add_argument('--samplePerSNR', default=10, type=int, help='Number of samples per SNR')
    parser.add_argument('--whichBest', default="bestValLoss", type=str, help='Best of acc or loss')
    parser.add_argument('--Ncues', default=5, type=int, help='Number of cues')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--prepMethod', default="None", type=str, help='Preprocessing method')
    parser.add_argument('--isSave', default="True", type=str, help='Save the plots?')


    args = parser.parse_args()
    if args.audioDir[-1] != "/":
        args.audioDir += "/"
    if args.hrirDir[-1] != "/":
        args.hrirDir += "/"
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
    if args.isSave == "True":
        args.isSave = True
    else:
        args.isSave = False

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
    args.valSNRList = [float(item) for item in args.valSNRList.split(',')]
    print("Range of SNR: ", args.valSNRList)
    print("Number of samples per SNR: ", args.samplePerSNR)
    print("Number of cues: ", args.Ncues)
    print("Preprocessing method: ", args.prepMethod)

    path = args.hrirDir + "/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    path = glob(os.path.join(args.audioDir+"/*"))
    Naudio = len(path)
    print("Number of audio files: ", Naudio)

    cuesShape = CuesShape(Ncues=args.Ncues, valSNRList=args.valSNRList)
    lenSliceInSec = cuesShape.lenSliceInSec
    Nfreq = cuesShape.Nfreq
    Ntime = cuesShape.Ntime
    Ncues = cuesShape.Ncues
    Nloc = cuesShape.Nloc
    Nsample = Nloc * args.samplePerSNR
    valSNRList = cuesShape.valSNRList
    preprocess = Preprocess(prep_method=args.prepMethod)

    # allocate tensors cues and labels in RAM
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    if args.task == "allRegression":
        labels_ = torch.zeros((Nsample,3), dtype=torch.float32)
    elif args.task == "multisound":
        labels_ = torch.zeros((Nsample,2), dtype=torch.float32)
    else:
        labels_ = torch.zeros((Nsample,))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.whichModel.lower() == "transformer":
        model = DIYModel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    elif args.whichModel.lower() == "paralleltransformer":
        model = DIY_parallel(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    elif args.whichModel.lower() == "cnn":
        model = CNNModel(task=args.task, Ncues=Ncues, dropout=args.valDropout, device=device, isDebug=args.isDebug)
    elif args.whichModel.lower() == "pytorchtransformer":
        model = PytorchTransformer(args.task, Ntime, Nfreq, Ncues, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    else:
        raise SystemExit("No model selected")

    if args.isHPC:
        model = nn.DataParallel(model)
    model = model.to(device)

    # learning_rate = 1e-4
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    model, val_optim = loadCheckpoint(model=model, optimizer=None, scheduler=None, loadPath=args.modelDir, task=args.task, phase="test", whichBest=args.whichBest)

    UD_confusion = []
    LR_confusion = []
    FB_confusion = []
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

                    ipdCues = preprocess(calIPD(specLeft, specRight))
                    ildCues = preprocess(calILD(specLeft, specRight))
                    r_l, theta_l  = cartesian2euler(specLeft)
                    r_r, theta_r  = cartesian2euler(specRight)
                    r_l = preprocess(r_l)
                    theta_l = preprocess(theta_l)
                    r_r = preprocess(r_r)
                    theta_r = preprocess(theta_r)
                    
                    if Ncues == 6:
                        cues = concatCues([ipdCues, ildCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                    elif Ncues == 5:
                        cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                    elif Ncues == 4:
                        cues = concatCues([r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                    elif Ncues == 2:
                        cues = concatCues([ipdCues, ildCues], (Nfreq, Ntime))

                    cues_[fileCount] = cues
                    if args.task == "allRegression":
                        labels_[fileCount][0] = locIndex
                        labels_[fileCount][1:3] = locIndex2Label(locLabel, locIndex, args.task)
                    elif args.task == "multisound":
                        labels_[fileCount] = torch.tensor(
                            [
                                locIndex2Label(locLabel, locIndex, "elevRegression"),
                                locIndex2Label(locLabel, locIndex, "azimRegression"),
                            ]
                        )
                    else:
                        labels_[fileCount] = locIndex2Label(locLabel, locIndex, args.task)

                    fileCount += 1
                    # if fileCount % (Nloc*len(valSNRList)) == 0:
                    #     print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                    #           fileCount // (Nloc*len(valSNRList)))

        # create tensor dataset from data loaded in RAM
        if args.task in ["allRegression", "multisound"]:
            dataset = TensorDataset(cues_, labels_)
        else:
            dataset = TensorDataset(cues_, labels_.long())

        test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=args.numWorker)

        # test phase
        
        if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
            # confusion matrix
            confusion_matrix = torch.zeros(predNeuron(args.task), predNeuron(args.task))

        test_correct = 0.0
        test_total = 0.0
        test_sum_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        
        confusion = ConfusionEval(Nsample, savePath = args.modelDir, expName="SNR="+str(int(valSNR)), isSave=args.isSave)
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                outputs = model(inputs)

                if args.task in ["elevRegression","azimRegression","allRegression"]:
                    loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels[:, 1:3]))))
                
                    confusion.evaluate(outputs, labels[:, 1:3])
                elif args.task == "multisound":
                    loss = torch.sqrt(torch.mean(torch.square(DoALoss(outputs, labels))))
                else:
                    loss = nn.CrossEntropyLoss(outputs, labels)
                test_sum_loss += loss.item()

                if not (args.task in ["elevRegression","azimRegression","allRegression","multisound"]):
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.data).sum().item()

                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                # else:
                    # test_total += labels.shape[0]
                    # test_correct += regressionAcc(outputs, labels, locLabel, device)
        test_loss = test_sum_loss / (i+1)
        if args.task in ["elevRegression","azimRegression","allRegression","multisound"]:
            test_acc = radian2Degree(test_loss)
            print('For SNR: %d Test Loss: %.04f | RMS angle (degree): %.04f '
                % (valSNR, test_loss, test_acc))
        else:
            test_acc = round(100.0 * test_correct / test_total, 2)
            print('For SNR: %d Test Loss: %.04f | Test Acc: %.4f%% '
                % (valSNR, test_loss, test_acc))
        print("outputs:", outputs[0:10])
        print("labels:", labels[0:10])
                
        out = confusion.report()
        UD_confusion.append(out[0])
        LR_confusion.append(out[1])
        FB_confusion.append(out[2])
    plotConfusion(
        valSNRList,
        UD_confusion,
        LR_confusion,
        FB_confusion,
        savePath=args.modelDir,
        isSave=args.isSave
    )