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

from load_data import *
from utils import *
from model_transformer import *
from loss import *
from main_train import *
   
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing hyperparamters')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('modelDir', type=str, help='Directory of saved model')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--samplePerSNR', default=100, type=int, help='Number of samples per SNR')
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

    lenSliceInSec = 0.5   # length of audio slice in sec
    Nfreq = 512
    Ntime = 44
    Ncues = 5
    Nloc = 187
    Nsample = Nloc * args.samplePerSNR

    # allocate tensors cues and labels in RAM
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    labels_ = torch.zeros((Nsample,))

    valSNRList = [-10,-5,0,5,10,15,20,25,100]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FC3(args.task, Ntime, Nfreq, Ncues, args.numEnc, 8, device, 4, args.valDropout, args.isDebug).to(device)
    learning_rate = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    model, optimizer, scheduler, pretrainEpoch, val_acc_optim = loadCheckpoint(model, optimizer, scheduler, args.modelDir, args.task, "test")

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
                    labels_[fileCount] = locIndex2Label(locLabel, locIndex, args.task)

                    fileCount += 1
                    # if fileCount % (Nloc*len(valSNRList)) == 0:
                    #     print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                    #           fileCount // (Nloc*len(valSNRList)))

        # create tensor dataset from data loaded in RAM
        dataset = TensorDataset(cues_, labels_.long())

        test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=args.numWorker)

        # testing
        
        if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
            # confusion matrix
            confusion_matrix = torch.zeros(predNeuron(args.task), predNeuron(args.task))

        test_correct = 0.0
        test_total = 0.0
        test_sum_loss = 0.0
        test_loss = 0.0
        test_acc = 0.0
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                test_sum_loss += loss.item()
                if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.data).sum().item()

                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                # else:
                    # test_total += labels.shape[0]
                    # test_correct += regressionAcc(outputs, labels, locLabel, device)
        test_loss = test_sum_loss / (i+1)
        test_acc = round(100.0 * test_correct / test_total, 2)
        print('For SNR: %d Test_Loss: %.04f | Test_Acc: %.4f%% '
            % (valSNR, test_loss, test_acc))

'''
###################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing hyperparamters')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('modelDir', type=str, help='Directory of saved model')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
    parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
    parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size')
    parser.add_argument('--samplePerSNR', default=100, type=int, help='Number of samples per SNR')
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

    lenSliceInSec = 0.5   # length of audio slice in sec
    Nfreq = 512
    Ntime = 44
    Ncues = 5
    Nloc = 187
    Nsample = Nloc * args.samplePerSNR

    # allocate tensors cues and labels in RAM
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    labels_ = torch.zeros((Nsample,))

    valSNRList = [-10,-5,0,5,10,15,20,25,100]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FC3(args.task, Ntime, Nfreq, Ncues, args.numEnc, 8, device, 4, args.valDropout, args.isDebug).to(device)
    model, val_optim = loadCheckpoint(model, optimizer, scheduler, args.modelDir, args.task, "test")
    print("Retrieved the model at epoch: ", pretrainEpoch)

    

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
                    labels_[fileCount] = locIndex2Label(locLabel, locIndex, args.task)

                    '''if fileCount == 23:
                        raise SystemExit("Debugging")'''

                    fileCount += 1
                    # if fileCount % (Nloc*len(valSNRList)) == 0:
                    #     print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                    #           fileCount // (Nloc*len(valSNRList)))

        # create tensor dataset from data loaded in RAM
        dataset = TensorDataset(cues_, labels_.long())

        test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=args.numWorker)

        '''
        testing
        '''
        if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
            # confusion matrix
            confusion_matrix = torch.zeros(predNeuron(args.task), predNeuron(args.task))

        test_correct = 0.0
        test_total = 0.0
        test_sum_loss = 0.0
        test_loss = 0.0
        train_acc = 0.0
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                # print(inputs.shape)

                outputs = model(inputs)
                # print(outputs.shape)
                # print(labels.shape)
                test_sum_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels.data).sum().item()
                
                if args.task == "elevClass" or args.task == "azimClass" or args.task == "allClass":
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        test_loss = test_sum_loss / (i+1)
        test_acc = round(100.0 * test_correct / test_total, 2)
        print('For SNR: %d Test_Loss: %.04f | Test_Acc: %.4f%% '
            % (valSNR, test_loss, test_acc))