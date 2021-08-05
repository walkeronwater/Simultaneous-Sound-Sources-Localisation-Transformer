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

'''
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
                            # mix only the different locations at the same elevation
                            elif locLabel[locIndex_1, 0] != locLabel[locIndex_2, 0]:
                                continue

                            sigLeft_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 0])
                            sigRight_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 1])

                            # print("Location index: ", locIndex)
                            # showSpectrogram(sigLeft, fs_HRIR)
                            # showSpectrogram(sigRight, fs_HRIR)
                            specLeft = calSpectrogram(sigLeft_1 + sigLeft_2, fs_audio)
                            specRight = calSpectrogram(sigRight_1 + sigRight_2, fs_audio)

                            ipdCues = preprocess(calIPD(specLeft, specRight))
                            r_l, theta_l  = cartesian2euler(specLeft)
                            r_r, theta_r  = cartesian2euler(specRight)
                            r_l = preprocess(r_l)
                            theta_l = preprocess(theta_l)
                            r_r = preprocess(r_r)
                            theta_r = preprocess(theta_r)

                            if Ncues == 6:
                                ildCues = preprocess(calILD(specLeft, specRight))
                                cues = concatCues([ipdCues, ildCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                            elif Ncues == 5:
                                cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                            elif Ncues == 4:
                                cues = concatCues([r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))
                            elif Ncues == 2:
                                ildCues = preprocess(calILD(specLeft, specRight))
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

# @jit(nopython=True)
# def conv(seq1, seq2):
#     return np.convolve(seq1, seq2)


def createCues_(
    # path,
    src_1, src_2, Nsample, cuesShape, prep_method, dirName, locLabel
):
    cuesShape = CuesShape()
    Nfreq = cuesShape.Nfreq
    Ntime = cuesShape.Ntime
    Ncues = cuesShape.Ncues
    Nloc = cuesShape.Nloc
    lenSliceInSec = cuesShape.lenSliceInSec
    binaural_cues = BinauralCues(prep_method=prep_method, fs_audio=fs_audio)
    save_cues = SaveCues(savePath=dirName, locLabel=locLabel)

    for audioIndex_1 in range(len(src_1)):
        print("Audio 1 index: ", audioIndex_1)
        audio_1, fs_audio_1 = sf.read(src_1[audioIndex_1])
        #[TODO] change fs_HRIR to fs_audio
        audioSliceList_1 = audioSliceGenerator(audio_1, fs_HRIR, lenSliceInSec, threshold=0)
        print(len(audioSliceList_1))
        
        # loop 2: audio j from 1 -> Naudio but skip when i==j
        for audioIndex_2 in range(audioIndex_1+1, len(src_2)):
            print("Audio 2 index: ", audioIndex_2)
            audio_2, fs_audio_2 = sf.read(src_2[audioIndex_2])
            #[TODO] change fs_HRIR to fs_audio
            audioSliceList_2 = audioSliceGenerator(audio_2, fs_HRIR, lenSliceInSec, threshold=0)
            print(len(audioSliceList_2))

            # loop 3: audioSlice of audio i from 1 -> NaudioSlice
            for sliceIndex_1 in range(len(audioSliceList_1)):
                audioSlice_1 = audio_1[audioSliceList_1[sliceIndex_1]]
        
                # loop 4: audioSlice of audio j from 1 -> NaudioSlice
                for sliceIndex_2 in range(len(audioSliceList_2)):
                    audioSlice_2 = audio_2[audioSliceList_2[sliceIndex_2]]

                    # loop 5: loc of slice of audio 1 from 0 to 186
                    for locIndex_1 in loc_region.low_left + loc_region.high_left:
                        sigLeft_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 0])
                        sigRight_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 1])
                        # sigLeft_1 = conv(audioSlice_1, hrirSet_re[locIndex_1, 0])
                        # sigRight_1 = conv(audioSlice_1, hrirSet_re[locIndex_1, 1])

                        # loop 6: loc of slice of audio 1 from 5 to 186 (not adjacent locs)
                        for locIndex_2 in loc_region.low_right + loc_region.high_right:
                            # print(audioIndex_1, sliceIndex_1, locIndex_1)
                            # print(audioIndex_2, sliceIndex_2, locIndex_2)
                            # region_1 = loc_region.whichRegion(locIndex_1)
                            # region_2 = loc_region.whichRegion(locIndex_2)
                            # if region_1[-1] == region_2[-1]:
                            #     continue
                            # elif region_1 == "None" or region_2 == "None":
                            #     continue
                            # mix only the different locations at the same elevation
                            # elif locLabel[locIndex_1, 0] != locLabel[locIndex_2, 0]:
                            #     continue
                            
                            sigLeft_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 0])
                            sigRight_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 1])
                            # sigLeft_2 = conv(audioSlice_2, hrirSet_re[locIndex_2, 0])
                            # sigRight_2 = conv(audioSlice_2, hrirSet_re[locIndex_2, 1])

                            ipd, magL, phaseL, magR, phaseR = binaural_cues(sigLeft_1+sigLeft_2, sigRight_1+sigRight_2)
                            save_cues([ipd, magL, phaseL, magR, phaseR], [locIndex_1, locIndex_2])
                            if save_cues.fileCount == Nsample:
                                return

def createTrainingSet():
    timeFlag = True
    hrirSet, locLabel, fs_HRIR = loadHRIR(args.hrirDir + "/IRC*")
    src_1_path = glob(os.path.join(args.trainAudioDir+"/speech_male/*"))
    src_2_path = glob(os.path.join(args.trainAudioDir+"/speech_female/*"))
    
    train_src_1_count = 0
    train_src_2_count = 0
    for i in range(len(src_1_path)):
        train_src_1 = AudioSignal(path=src_1_path[i], slice_duration=1)
        train_src_1_count += len(train_src_1.slice_list)
    train_src_2_count = 0
    for i in range(len(src_2_path)):
        train_src_2 = AudioSignal(path=src_2_path[i], slice_duration=1)
        train_src_2_count += len(train_src_2.slice_list)
    
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    print(hrirSet.shape)

    # audio indexes
    
    audio_index_1 = 0
    audio_index_2 = 0
    train_src_1 = AudioSignal(path=src_1_path[audio_index_1], slice_duration=1)
    train_src_2 = AudioSignal(path=src_2_path[audio_index_2], slice_duration=1)
    binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=train_src_1.fs_audio)
    loc_region = LocRegion(locLabel=locLabel)
    binaural_cues = BinauralCues(fs_audio=train_src_1.fs_audio, prep_method="standardise")
    save_cues = SaveCues(savePath=args.cuesDir+"/", locLabel=locLabel)
    
    start_time = time.time()
    slice_idx_1 = 0
    slice_idx_2 = 0
    count = 0
    while True:
        print(f"Current audio (src 1): {audio_index_1}, and (src 2): {audio_index_2}")
        # print(f"Number of slices (audio 1): {len(train_src_1.slice_list)}, and (audio 2): {len(train_src_2.slice_list)}")
        if slice_idx_1 >= len(train_src_1.slice_list):
            slice_idx_1 = 0
            audio_index_1 += 1
            train_src_1 = AudioSignal(path=train_src_1_path[audio_index_1], slice_duration=1)
            
        if slice_idx_2 >= len(train_src_2.slice_list):
            slice_idx_2 = 0
            audio_index_2 += 1
            train_src_2 = AudioSignal(path=train_src_2_path[audio_index_2], slice_duration=1)
        

        sig_sliced_1 = train_src_1(idx=slice_idx_1)
        sig_sliced_2 = train_src_2(idx=slice_idx_2)

        for loc_idx_1 in loc_region.high_left + loc_region.low_left:
            for loc_idx_2 in loc_region.high_right + loc_region.low_right:
                sigL_1, sigR_1 = binaural_sig(sig_sliced_1, loc_idx_1)
                sigL_2, sigR_2 = binaural_sig(sig_sliced_2, loc_idx_2)
                magL, phaseL, magR, phaseR = binaural_cues(sigL_1+sigL_2, sigR_1+sigR_2)

                save_cues(cuesList=[magL, phaseL, magR, phaseR], locIndex=[loc_idx_1, loc_idx_2])
                if save_cues.fileCount == args.Nsample:
                    return
            if timeFlag:
                print(f"One location loop costs {(time.time()-start_time)} seconds.")
                timeFlag = False
        slice_idx_1 += 1
        slice_idx_2 += 1
        count += 1
        # print(count)
        if count >= train_src_1_count or count >= train_src_2_count:
            return

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
    parser.add_argument('--prepMethod', default="normalise", type=str, help='Preprocessing method')
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
    
    if not os.path.isdir(args.cuesDir):
        os.mkdir(args.cuesDir)
    #     os.mkdir(args.cuesDir+"/train/")
    #     os.mkdir(args.cuesDir+"/valid/")
    # if not os.path.isdir(args.cuesDir+"/train/"):
    #     os.mkdir(args.cuesDir+"/train/")
    # if not os.path.isdir(args.cuesDir+"/valid/"):
    #     os.mkdir(args.cuesDir+"/valid/")
    
    createTrainingSet()
    