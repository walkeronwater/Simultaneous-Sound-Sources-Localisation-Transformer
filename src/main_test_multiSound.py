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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing hyperparamters')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('modelDir', type=str, help='Directory of saved model')
    parser.add_argument('numWorker', type=int, help='Number of workers')
    parser.add_argument('task', type=str, help='Task')
    parser.add_argument('whichModel', type=str, help='whichModel')
    parser.add_argument('Nsound', type=int, help='Number of sound')
    parser.add_argument('Nsample', type=int, help='Number of samples')
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
    print("Number of sound: ", args.Nsound)
    print("Number of samples: ", args.Nsample)
    
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
    Nsound = args.Nsound
    Nsample = args.Nsample
    valSNRList = cuesShape.valSNRList
    preprocess = Preprocess(prep_method=args.prepMethod)
    
    # resampling the HRIR
    _, fs_audio = sf.read(path[0])
    temp = librosa.resample(hrirSet[0, 0], fs_HRIR, fs_audio)
    hrirSet_re = np.empty(hrirSet.shape[0:2]+temp.shape)
    for i in range(hrirSet.shape[0]):
        hrirSet_re[i, 0] = librosa.resample(hrirSet[i, 0], fs_HRIR, fs_audio)
        hrirSet_re[i, 1] = librosa.resample(hrirSet[i, 1], fs_HRIR, fs_audio)
    print(hrirSet_re.shape)
    del temp, hrirSet

    loc_region = LocRegion(locLabel)
    
    # allocate tensors cues and labels in RAM
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    if args.task == "allRegression":
        labels_ = torch.zeros((Nsample,2*args.Nsound), dtype=torch.float32)
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
    elif args.whichModel.lower() == "multisound":
        model = DIY_multiSound(args.task, Ntime, Nfreq, Ncues, Nsound, args.numEnc, args.numFC, 8, device, 4, args.valDropout, args.isDebug)
    else:
        raise SystemExit("No model selected")

    if args.isHPC:
        model = nn.DataParallel(model)
    model = model.to(device)
    model, val_optim = loadCheckpoint(model=model, optimizer=None, scheduler=None, loadPath=args.modelDir, task=args.task, phase="test", whichBest=args.whichBest)
    
    fileCount = 0
    # loop 1: audio i from 1 -> Naudio
    for audioIndex_1 in range(len(path)):
        if fileCount == Nsample:
            break
        print("Audio 1 index: ", audioIndex_1)
        audio_1, fs_audio_1 = sf.read(path[audioIndex_1])
        # audio = librosa.resample(audio, fs_audio, fs_HRIR)

        #[TODO] change fs_HRIR to fs_audio
        audioSliceList_1 = audioSliceGenerator(audio_1, fs_HRIR, lenSliceInSec)
        
        # loop 2: audioSlice of audio i from 1 -> NaudioSlice
        for sliceIndex_1 in range(len(audioSliceList_1)):
            if fileCount == Nsample:
                break
            audioSlice_1 = audio_1[audioSliceList_1[sliceIndex_1]]

            # loop 3: audio j from 1 -> Naudio but skip when i==j
            for audioIndex_2 in range(len(path)):
                if fileCount == Nsample:
                    break
                if audioIndex_2 == audioIndex_1:
                    continue
                print("Audio 2 index: ", audioIndex_2)
                audio_2, fs_audio_2 = sf.read(path[audioIndex_2])
                #[TODO] change fs_HRIR to fs_audio
                audioSliceList_2 = audioSliceGenerator(audio_2, fs_HRIR, lenSliceInSec)

                # loop 4: audioSlice of audio j from 1 -> NaudioSlice
                for sliceIndex_2 in range(len(audioSliceList_2)):
                    if fileCount == Nsample:
                        break
                    audioSlice_2 = audio_2[audioSliceList_2[sliceIndex_2]]

                    # loop 5: loc of slice of audio 1 from 0 to 186
                    for locIndex_1 in range(Nloc):
                        if fileCount == Nsample:
                            break

                        # hrirLeft_re = librosa.resample(hrirSet[locIndex_1, 0], fs_HRIR, fs_audio)
                        # hrirRight_re = librosa.resample(hrirSet[locIndex_1, 1], fs_HRIR, fs_audio)
                        sigLeft_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 0])
                        sigRight_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 1])

                        # loop 6: loc of slice of audio 1 from 5 to 186 (not adjacent locs)
                        for locIndex_2 in range(Nloc):
                            if fileCount == Nsample:
                                break
                            if locIndex_2 == locIndex_1 or loc_region.whichRegion(locIndex_1) == loc_region.whichRegion(locIndex_2):
                                continue

                            sigLeft_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_1, 0])
                            sigRight_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_1, 1])

                            # print("Location index: ", locIndex)
                            # showSpectrogram(sigLeft, fs_HRIR)
                            # showSpectrogram(sigRight, fs_HRIR)
                            specLeft = calSpectrogram(sigLeft_1 + sigLeft_2)
                            specRight = calSpectrogram(sigRight_1 + sigRight_2)

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
                                labels_[fileCount] = torch.tensor(
                                    [
                                        locIndex2Label(locLabel, locIndex_1, "elevRegression"),
                                        locIndex2Label(locLabel, locIndex_1, "azimRegression"),
                                        locIndex2Label(locLabel, locIndex_2, "elevRegression"),
                                        locIndex2Label(locLabel, locIndex_2, "azimRegression")
                                    ]
                                )

                            fileCount += 1

    dataset = TensorDataset(cues_, labels_)
    test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=args.numWorker)

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
            print("inputs:", inputs.shape)
            print("outputs:", outputs)
            print("labels:", labels)
            loss = torch.sqrt(torch.mean(torch.square(cost_multiSound(outputs, labels))))
            test_sum_loss += loss.item()

            if not (args.task in ["elevRegression","azimRegression","allRegression"]):
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels.data).sum().item()
            raise SystemExit("debug")

    test_loss = test_sum_loss / (i+1)
    if args.task == "elevRegression" or args.task == "azimRegression" or args.task == "allRegression":
        test_acc = radian2Degree(test_loss)
        print('Test Loss: %.04f | RMS angle (degree): %.04f '
            % (test_loss, test_acc))