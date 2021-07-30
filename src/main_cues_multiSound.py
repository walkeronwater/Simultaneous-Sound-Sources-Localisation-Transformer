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
from numba import jit, njit
from numba.experimental import jitclass
import time

from load_data import *
from utils import *

#[TODO] hardcoded tensor size
class CuesShape:
    def __init__(
        self,
        Nfreq = 512,
        Ntime = 44,
        Ncues = 5,
        Nloc = 187,
        lenSliceInSec = 0.5,
        valSNRList = [-5,0,5,10,15,20,25,30,35]
    ):
        self.Nfreq = Nfreq
        self.Ntime = Ntime
        self.Ncues = Ncues
        self.Nloc = Nloc
        self.lenSliceInSec = lenSliceInSec
        self.valSNRList = valSNRList

#[TODO]
def mixLoc():
    pass

def createCues_multiSound(path, Nsample, cuesShape, prep_method, dirName):
    Nfreq = cuesShape.Nfreq
    Ntime = cuesShape.Ntime
    Ncues = cuesShape.Ncues
    Nloc = cuesShape.Nloc
    lenSliceInSec = cuesShape.lenSliceInSec
    valSNRList = cuesShape.valSNRList
    preprocess = Preprocess(prep_method=prep_method)
    fileCount = 0
    print("Creating cues in ", dirName)

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
                            region_1 = loc_region.whichRegion(locIndex_1)
                            region_2 = loc_region.whichRegion(locIndex_2)
                            if region_1[-1] == region_2[-1]:
                                continue
                            elif region_1 == "None" or region_2 == "None":
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

                            saveCues_multiSound(cues, locIndex_1, locIndex_2, dirName, fileCount, locLabel)

                            fileCount += 1

def saveCues_multiSound(cues, locIndex_1, locIndex_2, dirName, fileCount, locLabel):
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
            csvFile.write(str(locIndex2Label(locLabel, locIndex_1, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_1, "azimRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_2, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_2, "azimRegression")))
            csvFile.write('\n')
    else:
        with open(dirName+'dataLabels.csv', 'a') as csvFile:
            csvFile.write(str(fileCount))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_1, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_1, "azimRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_2, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex_2, "azimRegression")))
            csvFile.write('\n')
    torch.save(cues, dirName+str(fileCount)+'.pt')

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
    parser.add_argument('Nsound', type=int, help='Number of sound')
    parser.add_argument('Nsample', type=int, help='Number of samples')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--valSNRList', default="-5,0,5,10,15,20,25,30,35", type=str, help='Range of SNR')
    parser.add_argument('--Ncues', default=5, type=int, help='Number of cues?')
    parser.add_argument('--prepMethod', default="None", type=str, help='Preprocessing method')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')

    args = parser.parse_args()
    print("Training audio files directory: ", args.trainAudioDir)
    print("Validation audio files directory: ", args.validAudioDir)
    print("HRIR files directory: ", args.hrirDir)
    args.trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
    print("Train validation split: ", args.trainValidSplit)
    args.valSNRList = [float(item) for item in args.valSNRList.split(',')]
    print("Range of SNR: ", args.valSNRList)
    print("Number of cues: ", args.Ncues)
    print("Preprocessing method: ", args.prepMethod)
    print("Number of sound: ", args.Nsound)
    print("Number of samples: ", args.Nsample)

    Nsample_train = int(args.trainValidSplit[0]*args.Nsample)
    Nsample_valid = int(args.trainValidSplit[1]*args.Nsample)
    print("Training data volume: ", Nsample_train)
    print("Validation data volume: ", Nsample_valid)

    hrirSet, locLabel, fs_HRIR = loadHRIR(args.hrirDir + "/IRC*")
    trainAudioPath = glob(os.path.join(args.trainAudioDir+"/*"))
    validAudioPath = glob(os.path.join(args.validAudioDir+"/*"))
    print("Number of training audio files: ", len(trainAudioPath))
    print("Number of validation audio files: ", len(validAudioPath))

    cuesShape = CuesShape(valSNRList=args.valSNRList, Ncues=args.Ncues)
    # print(cuesShape.valSNRList)
    # raise SystemExit('debug')

    # resampling the HRIR
    _, fs_audio = sf.read(trainAudioPath[0])
    temp = librosa.resample(hrirSet[0, 0], fs_HRIR, fs_audio)
    hrirSet_re = np.empty(hrirSet.shape[0:2]+temp.shape)
    for i in range(hrirSet.shape[0]):
        hrirSet_re[i, 0] = librosa.resample(hrirSet[i, 0], fs_HRIR, fs_audio)
        hrirSet_re[i, 1] = librosa.resample(hrirSet[i, 1], fs_HRIR, fs_audio)
    print(hrirSet_re.shape)
    del temp, hrirSet

    loc_region = LocRegion(locLabel)
    
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

    start_time = time.time()
    createCues_multiSound(trainAudioPath, Nsample_train, cuesShape, args.prepMethod, dirName=args.cuesDir+"/train/")
    print("Time elapsed: ", time.time()-start_time)
    print(validAudioPath)
    createCues_multiSound(validAudioPath, Nsample_valid, cuesShape, args.prepMethod, dirName=args.cuesDir+"/valid/")


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
