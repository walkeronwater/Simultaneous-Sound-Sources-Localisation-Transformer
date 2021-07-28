from math import pi
import soundfile as sf
from scipy import signal
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
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

# method to generate audio slices for a given length requirement
# with a hardcoded power threshold
def audioSliceGenerator(audioSeq, sampleRate, lenSliceInSec, isDebug=False):
    lenAudio = audioSeq.size
    lenSlice = round(sampleRate * lenSliceInSec)
    # audioSliceList = [range(lenSlice*i, lenSlice *(i+1)) for i in range(lenAudio//lenSlice)]
    # print(len(audioSliceList))
    # print(lenAudio//lenSlice)

    audioSliceList = []
    if isDebug:
        powerList = []
    # threshold for spectrum power
    for i in range(lenAudio//lenSlice):
        sliced = audioSeq[lenSlice*i:lenSlice *(i+1)]
        # print("slice power", np.mean(np.power(sliced, 2)))
        if isDebug:
            powerList.append(np.mean(np.power(sliced, 2)))
        if np.mean(np.power(sliced, 2)) > 0.01:
            audioSliceList.append(range(lenSlice*i, lenSlice *(i+1)))
    if isDebug:
        return audioSliceList, powerList
    else:
        return audioSliceList

# method to generate a sequence of noise for a given SNR
def noiseGenerator(sigSeq, valSNR):
    # assert debug
    # assert (
    #     valSNR >= 20
    # ), "Input data needs to be reshaped to (1, length of sequence)"
    if valSNR >= 100:
        return 0
    else:
        sigSeqPower = 10*np.log10(np.mean(np.power(sigSeq, 2)))
        noiseSeqPower = np.power(10, (sigSeqPower - valSNR)/10)
        noiseSeq = np.random.normal(0, np.sqrt(noiseSeqPower), sigSeq.shape)
        del sigSeqPower, noiseSeqPower
        return noiseSeq

'''def addNoise(sigPair):
    valSNR = 1
    # loop through all training examples
    for i in range(sigPairList[0].shape[0]):
        # loop through all locations
        for locIndex in range(sigPairList[0].shape[1]):
            noiseLeft = noiseGenerator(np.expand_dims(sigPairList[0][i,locIndex,0], axis=0), valSNR)
            noiseRight = noiseGenerator(np.expand_dims(sigPairList[0][i,locIndex,1], axis=0), valSNR)'''

# utility methods for binaural cue extraction
def cartesian2euler(seq):
    x = seq.real
    y = seq.imag

    r = np.sqrt(x**2+y**2)
    theta = np.arctan(
        np.divide(y, x, where=x!=0)
    )
    # if x != 0:
    #     theta = np.arctan(y/x)
    # else:
    #     theta = np.pi/2
        
    return r, theta

def calIPD(seqL, seqR):
    ipd = np.angle(np.divide(seqL, seqR, out=np.zeros_like(seqL), where=np.absolute(seqR)!=0))
    return ipd

def calILD(seqL, seqR):
    ild = 20*np.log10(np.divide(np.absolute(seqL), np.absolute(seqR), out=np.zeros_like(np.absolute(seqL)), where=np.absolute(seqR)!=0))
    return ild

# [TODO] method to normalise a sequence which can be broadcasted to a sequence of sequence
# min-max/standardise/L2 norm for each tensor like an image
class Preprocess:
    def __init__(self, prep_method: str):
        self.prep_method = prep_method
        if self.prep_method.lower() == "standardise":
            print("Preprocessing method: standardise")
        elif self.prep_method.lower() == "normalise":
            print("Preprocessing method: normalise")
        else:
            print("Preprocessing method: none")
    def __call__(self, seq):
        if self.prep_method.lower() == "standardise":
            return (seq - np.mean(seq))/(np.std(seq))
        elif self.prep_method.lower() == "normalise":
            return seq/np.linalg.norm(seq)
        else:
            return seq

# [TODO] method to log IPD cues and spectral cues
'''def cuesLog():'''

def calSpectrogram(seq):
    Zxx = librosa.stft(seq, 1023, hop_length=512)
    return Zxx

def concatCues(cuesList: list, cuesShape: tuple):
    lastDim = len(cuesList)
    cues = torch.zeros(cuesShape+(lastDim,), dtype=torch.float)

    for i in range(lastDim):
        cues[:,:,i] = torch.from_numpy(cuesList[i])

    return cues

def locIndex2Label(locLabel, locIndex, task):
    if task == "elevClass":
        labels = int(((locLabel[locIndex, 0]+45) % 150)/15)
    elif task == "azimClass":
        labels = int((locLabel[locIndex, 1] % 360)/15)
    elif task == "allClass":
        labels = int(locIndex)
    elif task == "elevRegression":
        labels = locLabel[locIndex, 0]/180.0*pi
    elif task == "azimRegression":
        labels = locLabel[locIndex, 1]/180.0*pi
    elif task == "allRegression":
        labels = torch.tensor(
            [
                locLabel[locIndex, 0]/180.0*pi,
                locLabel[locIndex, 1]/180.0*pi
            ], dtype=torch.float32
        )
    return labels

# save cues as pt files and write labels in a csv file
def saveCues(cues, locIndex, dirName, fileCount, locLabel):
    if fileCount == 0:
        if os.path.isfile(dirName+'dataLabels.csv'):
            print("Directory exists -- overwriting")
            # if input('Delete saved_cues? ') == 'y':
            #     print('ok')
            shutil.rmtree(dirName)
            os.mkdir(dirName)
        
        with open(dirName+'dataLabels.csv', 'w') as csvFile:
            csvFile.write(str(fileCount))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "allClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimRegression")))
            csvFile.write('\n')
    else:
        with open(dirName+'dataLabels.csv', 'a') as csvFile:
            csvFile.write(str(fileCount))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "allClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimRegression")))
            csvFile.write('\n')
    torch.save(cues, dirName+str(fileCount)+'.pt')

if __name__ == "__main__":
    # path = "./HRTF/IRC*"
    # hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    # print(locLabel/180.0*pi)
    # print(locIndex2Label(locLabel, 23, "azimRegression"))

    preprocess = Preprocess()
    a = np.arange(0, 9).reshape(3,3)
    print(Preprocess.normalise(a))