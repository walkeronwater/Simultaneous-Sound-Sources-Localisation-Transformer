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

from load_data import loadHRIR
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cues')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('cuesDir', type=str, help='Directory of cues to be saved')
    parser.add_argument('Nsample', type=int, help='Number of samples?')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    print("Audio files directory: ", args.audioDir)
    print("HRIR files directory: ", args.hrirDir)

    path = args.hrirDir + "/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    path = glob(os.path.join(args.audioDir+"/*"))
    Naudio = len(path)
    print("Number of audio files: ", Naudio)

    lenSliceInSec = 0.5   # length of audio slice in sec
    fileCount = 0   # count the number of data samples
    Nfreq = 512
    Ntime = 44
    Ncues = 5
    Nloc = 187
    Nsample = args.Nsample

    isDisk = True
    # allocate tensors cues and labels in RAM
    if not isDisk:
        if "cues_" in globals() or "cues_" in locals():
            del cues_
        if "labels_" in globals() or "labels_" in locals():
            del labels_
        cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
        labels_ = torch.zeros((Nsample, 1))

    valSNRList = [-10,-5,0,5,10,15,20,100]

    dirName = args.cuesDir

    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    print(path)
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

                    # save cues onto disk
                    if isDisk:
                        saveCues(cues, locIndex, dirName, fileCount, locLabel)
                    else:
                        cues_[fileCount] = cues
                        labels_[fileCount] = labels

                    '''if fileCount == 1:
                        raise SystemExit("Debugging")'''

                    fileCount += 1
                    if fileCount % (Nloc*len(valSNRList)) == 0:
                        print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                            fileCount // (Nloc*len(valSNRList)))
