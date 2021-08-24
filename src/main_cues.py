import time
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

from data_loader import *
from utils import *

def createTrainingSet():
    timeFlag = True
    load_hrir = LoadHRIR(path="./HRTF/IRC*")

    path_1 = glob(os.path.join(args.trainAudioDir+"/speech_male/*"))
    path_2 = glob(os.path.join(args.trainAudioDir+"/speech_female/*"))
    
    src_path = []
    flag = True
    i, j = 0, 0
    while i < len(path_1) and j < len(path_2):
        if flag:
            src_path.append(path_1[i])
            i += 1
            flag = False
        else:
            src_path.append(path_2[j])
            j += 1
            flag = True

    src_count = 0
    for i in range(len(src_path)):
        src = AudioSignal(path=src_path[i], slice_duration=1)
        src_count += len(src.slice_list)

    # audio indexes
    audio_index = 0
    src = AudioSignal(path=src_path[audio_index], slice_duration=1)
    binaural_sig = BinauralSignal(hrir=load_hrir.hrir_set, fs_hrir=load_hrir.fs_HRIR, fs_audio=src.fs_audio)
    binaural_cues = BinauralCues(fs_audio=src.fs_audio, prep_method=args.prepMethod)
    save_cues = SaveCues(savePath=args.cuesDir+"/", locLabel=load_hrir.loc_label)
    
    start_time = time.time()
    slice_idx = 0
    count = 0
    count_file = 0
    while True:
        if slice_idx >= len(src.slice_list):
            slice_idx = 0
            audio_index += 1
            src = AudioSignal(path=src_path[audio_index], slice_duration=1)

        print(f"Current audio: {src_path[audio_index]}")
        # print(f"Number of slices: {len(src.slice_list)}")
        
        sig_sliced = src(idx=slice_idx)

        print("locations: ",load_hrir.loc_label.shape[0])
        for loc_idx in range(load_hrir.loc_label.shape[0]):
            for val_SNR in args.valSNRList:
                binaural_sig.val_SNR = val_SNR

                sigL, sigR = binaural_sig(sig_sliced, loc_idx)
                magL, phaseL, magR, phaseR = binaural_cues(sigL, sigR)

                save_cues(cuesList=[magL, phaseL, magR, phaseR], loc_idx_list=[loc_idx])
                count_file += 1
                if count_file >= args.Nsample:
                    return

                if timeFlag:
                    print(f"One location loop costs {(time.time()-start_time)} seconds.")
                    timeFlag = False
        slice_idx += 1
        count += 1
        # print(count)
        if count >= src_count:
            return


"""
This will create two folders containing data for training, validation. Each dataset will be
balanced and the volume will be decided by the train-validation ratio and total amount given
by the user input.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cues')
    parser.add_argument('trainAudioDir', type=str, help='Directory of audio files for training')
    parser.add_argument('validAudioDir', type=str, help='Directory of audio files for validation')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('cuesDir', type=str, help='Directory of cues to be saved')
    parser.add_argument('Nsample', type=int, help='Number of samples?')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--valSNRList', default="-5,0,5,10,15,20,25,30,35", type=str, help='Range of SNR')
    parser.add_argument('--Ncues', default=4, type=int, help='Number of cues?')
    parser.add_argument('--prepMethod', default="minmax", type=str, help='Preprocessing method')
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

    Nsample_train = int(args.trainValidSplit[0]*args.Nsample)
    Nsample_valid = int(args.trainValidSplit[1]*args.Nsample)
    print("Training data volume: ", Nsample_train)
    print("Validation data volume: ", Nsample_valid)

    hrirSet, locLabel, fs_HRIR = loadHRIR(args.hrirDir + "/IRC*")

    trainAudioPath = glob(os.path.join(args.trainAudioDir+"/*"))
    validAudioPath = glob(os.path.join(args.validAudioDir+"/*"))
    print("Number of training audio files: ", len(trainAudioPath))
    print("Number of validation audio files: ", len(validAudioPath))

    # cuesShape = CuesShape(valSNRList=args.valSNRList, Ncues=args.Ncues)
    # print(cuesShape.valSNRList)
    # raise SystemExit('debug')

    if not os.path.isdir(args.cuesDir):
        os.mkdir(args.cuesDir)
    #     os.mkdir(dirName+"/train/")
    #     os.mkdir(dirName+"/valid/")
    # if not os.path.isdir(dirName+"/train/"):
    #     os.mkdir(dirName+"/train/")
    # if not os.path.isdir(dirName+"/valid/"):
    #     os.mkdir(dirName+"/valid/")
    # raise SystemExit('debug')

    createTrainingSet()

