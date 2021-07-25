import argparse
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

from load_data import *
from utils import *

#[TODO] hardcoded tensor size
class CuesShape:
    Nfreq = 512
    Ntime = 44
    Ncues = 5
    Nloc = 187
    lenSliceInSec = 0.5     # length of audio slice in sec
    valSNRList = [-5,0,5,10,15,20,25]

def createCues(path, Nsample, CuesShape, dirName):
    Nfreq = CuesShape.Nfreq
    Ntime = CuesShape.Ntime
    Ncues = CuesShape.Ncues
    Nloc = CuesShape.Nloc
    lenSliceInSec = CuesShape.lenSliceInSec
    valSNRList = CuesShape.valSNRList

    fileCount = 0
    print("Creating cues in ", dirName)
    for audioIndex in range(len(path)):
        if fileCount == Nsample:
            break
        print("Audio index: ", audioIndex)
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
                
                for valSNR in valSNRList:
                    if fileCount == Nsample:
                        break
                
                    specLeft = calSpectrogram(sigLeft + noiseGenerator(sigLeft, valSNR))
                    specRight = calSpectrogram(sigRight + noiseGenerator(sigRight, valSNR))

                    ipdCues = calIPD(specLeft, specRight)
                    ildCues = calILD(specLeft, specRight)
                    r_l, theta_l  = cartesian2euler(specLeft)
                    r_r, theta_r  = cartesian2euler(specRight)

                    cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))

                    saveCues(cues, locIndex, dirName, fileCount, locLabel)

                    # if fileCount == 1:
                    #     raise SystemExit("Debugging")

                    fileCount += 1
                    if fileCount % (Nloc*len(valSNRList)) == 0:
                        print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                            fileCount // (Nloc*len(valSNRList)))



'''
# This will create two folders containing data for training, validation. Each dataset will be
# balanced and the volume will be decided by the train-validation ratio and total amount given
# by the user input.
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cues')
    parser.add_argument('trainAudioDir', type=str, help='Directory of audio files for training')
    parser.add_argument('validAudioDir', type=str, help='Directory of audio files for validation')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('cuesDir', type=str, help='Directory of cues to be saved')
    parser.add_argument('Nsample', type=int, help='Number of samples?')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    print("Training audio files directory: ", args.trainAudioDir)
    print("Validation audio files directory: ", args.validAudioDir)
    print("HRIR files directory: ", args.hrirDir)
    args.trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
    print("Train validation split: ", args.trainValidSplit)


    Nsample_train = int(args.trainValidSplit[0]*args.Nsample)
    Nsample_valid = int(args.trainValidSplit[1]*args.Nsample)
    print("Training data volume: ", Nsample_train)
    print("Validation data volume: ", Nsample_valid)

    hrirSet, locLabel, fs_HRIR = loadHRIR(args.hrirDir + "/IRC*")
    trainAudioPath = glob(os.path.join(args.trainAudioDir+"/*"))
    validAudioPath = glob(os.path.join(args.validAudioDir+"/*"))
    print("Number of training audio files: ", len(trainAudioPath))
    print("Number of validation audio files: ", len(validAudioPath))


    # fileCount = 0   # count the number of data samples
    
    dirName = args.cuesDir

    if not os.path.isdir(dirName):
        os.mkdir(dirName)
        os.mkdir(dirName+"/train/")
        os.mkdir(dirName+"/valid/")
    if not os.path.isdir(dirName+"/train/"):
        os.mkdir(dirName+"/train/")
    if not os.path.isdir(dirName+"/valid/"):
        os.mkdir(dirName+"/valid/")
    # raise SystemExit('debug')

    print(trainAudioPath)
    createCues(trainAudioPath, Nsample_train, CuesShape, dirName=args.cuesDir+"/train/")
    print(validAudioPath)
    createCues(validAudioPath, Nsample_valid, CuesShape, dirName=args.cuesDir+"/valid/")


    '''
    for audioIndex in range(len(trainAudioPath)):
        if fileCount == Nsample_train:
            break
        print("Audio index: ", audioIndex)
        audio, fs_audio = sf.read(path[audioIndex])
        # audio = librosa.resample(audio, fs_audio, fs_HRIR)

        audioSliceList = audioSliceGenerator(audio, fs_HRIR, lenSliceInSec)

        for sliceIndex in range(len(audioSliceList)):
            if fileCount == Nsample_train:
                break
            audioSlice = audio[audioSliceList[sliceIndex]]

            for locIndex in range(Nloc):
                if fileCount == Nsample_train:
                    break

                hrirLeft_re = librosa.resample(hrirSet[locIndex, 0], fs_HRIR, fs_audio)
                hrirRight_re = librosa.resample(hrirSet[locIndex, 1], fs_HRIR, fs_audio)
                sigLeft = np.convolve(audioSlice, hrirLeft_re)
                sigRight = np.convolve(audioSlice, hrirRight_re)

                # print("Location index: ", locIndex)
                # showSpectrogram(sigLeft, fs_HRIR)
                # showSpectrogram(sigRight, fs_HRIR)
                
                for valSNR in valSNRList:
                    if fileCount == Nsample:
                        break
                
                    specLeft = calSpectrogram(sigLeft + noiseGenerator(sigLeft, valSNR))
                    specRight = calSpectrogram(sigRight + noiseGenerator(sigRight, valSNR))

                    ipdCues = calIPD(specLeft, specRight)
                    ildCues = calILD(specLeft, specRight)
                    r_l, theta_l  = cartesian2euler(specLeft)
                    r_r, theta_r  = cartesian2euler(specRight)

                    cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))

                    # save cues onto disk
                    if isDisk:
                        saveCues(cues, locIndex, dirName, fileCount, locLabel)
                    else:
                        cues_[fileCount] = cues
                        labels_[fileCount] = labels

                    # if fileCount == 1:
                    #     raise SystemExit("Debugging")

                    fileCount += 1
                    if fileCount % (Nloc*len(valSNRList)) == 0:
                        print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                            fileCount // (Nloc*len(valSNRList)))
    '''
