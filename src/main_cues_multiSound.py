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

from data_loader import *
from utils import *

# @jit(nopython=True)
# def conv(seq1, seq2):
#     return np.convolve(seq1, seq2)

def nextAudioSlicePair(slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2, flag):
    if slice_idx_1 >= len(src_1.slice_list)-2:
        audio_index_1 += 1
        slice_idx_1 = 0
        src_1 = AudioSignal(path=src_1_path[audio_index_1], slice_duration=args.frameDuration, filter_type=args.filterType)
    if slice_idx_2 >= len(src_2.slice_list)-2:
        audio_index_2 += 1
        slice_idx_2 = 0
        src_2 = AudioSignal(path=src_2_path[audio_index_2], slice_duration=args.frameDuration, filter_type=args.filterType)

    if flag == [1,0]:
        sig_sliced_1 = src_1(idx=slice_idx_1)
        sig_sliced_2 = src_2(idx=slice_idx_2)
    if flag == [0,1]:
        sig_sliced_1 = src_2(idx=slice_idx_2)
        sig_sliced_2 = src_1(idx=slice_idx_1)
    elif flag == [1,1]:
        sig_sliced_1 = src_1(idx=slice_idx_1)
        slice_idx_1 += 1
        sig_sliced_2 = src_1(idx=slice_idx_1)
    elif flag == [0,0]:
        sig_sliced_1 = src_2(idx=slice_idx_2)
        slice_idx_2 += 1
        sig_sliced_2 = src_2(idx=slice_idx_2)

    sig_sliced_1 = src_1.apply_gain(sig_sliced_1, target_power=-20)
    sig_sliced_2 = src_2.apply_gain(sig_sliced_2, target_power=-20)

    return sig_sliced_1, sig_sliced_2, slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2

def selectSlices():
    pass

def createTrainingSet(src_1, src_1_count, src_2, src_2_count):
    audio_index_1 = 0
    audio_index_2 = 0
    slice_idx_1 = 0
    slice_idx_2 = 0
    count = 0
    count_file = 0
    start_time = time.time()
    timeFlag = True
    while True:
        print(f"Current audio (src 1): {audio_index_1}, and (src 2): {audio_index_2}")
        print(f"Number of slices (audio 1): {len(src_1.slice_list)}, and (audio 2): {len(src_2.slice_list)}")
        print(f"Source indexes: {slice_idx_1, slice_idx_2}")
        for bias in range(0,4):
            if count%4 == 0:
                flag=[1,0]
            elif count%4 == 1:
                flag=[0,1]
            elif count%4 == 2:
                flag=[1,1]
            elif count%4 == 3:
                flag=[0,0]

            print(f"Flag: {flag}")
            sig_sliced_1, sig_sliced_2, slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2 = nextAudioSlicePair(
                slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2, flag=flag
            )

            for loc_1 in range(bias, load_hrir.hrir_set.shape[0], 2):
                for loc_2 in range(bias, load_hrir.hrir_set.shape[0], 2):
                    if getAngleDiff(load_hrir.loc_label[loc_1], load_hrir.loc_label[loc_2]) <= 30:
                        continue
                    
                    sigL_1, sigR_1 = binaural_sig(sig_sliced_1, loc_1)
                    sigL_2, sigR_2 = binaural_sig(sig_sliced_2, loc_2)
                    magL, phaseL, magR, phaseR = binaural_cues(sigL_1+sigL_2, sigR_1+sigR_2)
                    # print(f"magL shape: {magL.shape}")
                    save_cues(cuesList=[magL, phaseL, magR, phaseR], loc_idx_list=[loc_1, loc_2])
                    count_file += 1
                    if count_file >= args.Nsample:
                        return

                if timeFlag:
                    print(f"One location loop costs {(time.time()-start_time)} seconds.")
                    timeFlag = False
            slice_idx_1 += 1
            slice_idx_2 += 1
            count += 1
            print(f"{count, count_file}")
            if count >= src_1_count or count >= src_2_count or audio_index_1 >= len(src_1_path)-1 or audio_index_2 >= len(src_2_path)-1:
                return

def createTestSet(src_1, src_1_count, src_2, src_2_count):
    audio_index_1 = 0
    audio_index_2 = 0
    slice_idx_1 = 0
    slice_idx_2 = 0
    count = 0
    count_file = 0
    
    while True:
        print(f"Current audio (src 1): {audio_index_1}, and (src 2): {audio_index_2}")
        print(f"Number of slices (audio 1): {len(src_1.slice_list)}, and (audio 2): {len(src_2.slice_list)}")
        print(f"Source indexes: {slice_idx_1, slice_idx_2}")
        
        if args.mix.lower() == "mf": flag = [1,0]
        elif args.mix.lower() == "mm": flag = [1,1]
        elif args.mix.lower() == "ff": flag = [0,0]
        print(f"Flag: {flag}")
        sig_sliced_1, sig_sliced_2, slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2 = nextAudioSlicePair(
            slice_idx_1, audio_index_1, slice_idx_2, audio_index_2, src_1, src_2, flag=flag
        )

        for loc_1 in range(load_hrir.hrir_set.shape[0]):
            for loc_2 in range(load_hrir.hrir_set.shape[0]):
                if getAngleDiff(load_hrir.loc_label[loc_1], load_hrir.loc_label[loc_2]) <= 30:
                    continue
                
                if args.valSNR <100:
                    binaural_sig.val_SNR = args.valSNR
                
                sigL_1, sigR_1 = binaural_sig(sig_sliced_1, loc_1)
                sigL_2, sigR_2 = binaural_sig(sig_sliced_2, loc_2)
                magL, phaseL, magR, phaseR = binaural_cues(sigL_1+sigL_2, sigR_1+sigR_2)
                # print(f"magL shape: {magL.shape}")
                save_cues(cuesList=[magL, phaseL, magR, phaseR], loc_idx_list=[loc_1, loc_2])
                count_file += 1

        slice_idx_1 += 1
        slice_idx_2 += 1
        count += 1
        if count >= args.Nsample:
            return
        print(f"{count, count_file}")
        if count >= src_1_count or count >= src_2_count or audio_index_1 >= len(src_1_path)-1 or audio_index_2 >= len(src_2_path)-1:
            return


"""
This will create two folders containing data for training, validation. Each dataset will be
balanced and the volume will be decided by the train-validation ratio and total amount given
by the user input.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cues')
    parser.add_argument('src1Dir', type=str, help='Directory of audio files for source 1')
    parser.add_argument('src2Dir', type=str, help='Directory of audio files for source 2')
    parser.add_argument('hrirDir', type=str, help='Directory of HRIR files')
    parser.add_argument('cuesDir', type=str, help='Directory of cues to be saved')
    parser.add_argument('Nsound', type=int, help='Number of sound')
    parser.add_argument('Nsample', type=int, help='Number of samples')
    parser.add_argument('--frameDuration', default=1, type=float, help='Duration of frames')
    parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
    parser.add_argument('--valSNRList', default="-5,0,5,10,15,20,25,30,35", type=str, help='Range of SNR')
    parser.add_argument('--Ncues', default=4, type=int, help='Number of cues?')
    parser.add_argument('--prepMethod', default="minmax", type=str, help='Preprocessing method')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--job', default="train", type=str, help='Trainset or testset?')
    parser.add_argument('--mix', default="mf", type=str, help='Mixing strategy?')
    parser.add_argument('--valSNR', default=100, type=int, help='SNR value')
    parser.add_argument('--filterType', default="None", type=str, help='Filter')

    args = parser.parse_args()
    # print("Training audio files directory: ", args.trainAudioDir)
    # print("Validation audio files directory: ", args.validAudioDir)
    print("HRIR files directory: ", args.hrirDir)
    args.trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
    print("Train validation split: ", args.trainValidSplit)
    args.valSNRList = [float(item) for item in args.valSNRList.split(',')]
    print("Range of SNR: ", args.valSNRList)
    print("Number of cues: ", args.Ncues)
    print("Preprocessing method: ", args.prepMethod)
    print("Number of sound: ", args.Nsound)
    print("Number of samples: ", args.Nsample)
    if args.valSNR <100:
        print("Mixed with noise")
    Nsample_train = int(args.trainValidSplit[0]*args.Nsample)
    Nsample_valid = int(args.trainValidSplit[1]*args.Nsample)
    print("Training data volume: ", Nsample_train)
    print("Validation data volume: ", Nsample_valid)

    src_1_path = glob(os.path.join(args.src1Dir + "/*"))
    src_2_path = glob(os.path.join(args.src2Dir + "/*"))
    print(f"Total available number of audio files: {len(src_1_path)}, {len(src_2_path)}")

    """load HRIRs and audio files"""
    # hrirSet, locLabel, fs_HRIR = loadHRIR(args.hrirDir + "/IRC*")
    load_hrir = LoadHRIR(path="./HRTF/IRC*")

    src_1_count = 0
    src_2_count = 0
    for i in range(len(src_1_path)):
        src_1 = AudioSignal(path=src_1_path[i], slice_duration=args.frameDuration)
        src_1_count += len(src_1.slice_list)
    for i in range(len(src_2_path)):
        src_2 = AudioSignal(path=src_2_path[i], slice_duration=args.frameDuration)
        src_2_count += len(src_2.slice_list)
    print(f"Total available number of audio slices: {src_1_count}, {src_2_count}")
    
    # audio indexes
    audio_index_1 = 0
    audio_index_2 = 0
    """Instantiate binaural cue classes"""
    src_1 = AudioSignal(path=src_1_path[audio_index_1], slice_duration=args.frameDuration, filter_type=args.filterType)
    src_2 = AudioSignal(path=src_2_path[audio_index_2], slice_duration=args.frameDuration, filter_type=args.filterType)
    binaural_sig = BinauralSignal(hrir=load_hrir.hrir_set, fs_hrir=load_hrir.fs_HRIR, fs_audio=src_1.fs_audio)
    binaural_cues = BinauralCues(fs_audio=src_1.fs_audio, prep_method=args.prepMethod)
    save_cues = SaveCues(savePath=args.cuesDir+"/", locLabel=load_hrir.loc_label)
    
    """create paths if not exist"""
    if not os.path.isdir(args.cuesDir):
        os.mkdir(args.cuesDir)
    try:
        os.path.isdir(args.cuesDir+"/train")
    except:
        os.mkdir(args.cuesDir+"/train")
    try:
        os.path.isdir(args.cuesDir+"/valid")
    except:
        os.mkdir(args.cuesDir+"/valid")

    if args.job.lower() == "train":
        createTrainingSet(
            src_1, src_1_count,src_2, src_2_count
        )
    elif args.job.lower() == "test":
        createTestSet(
            src_1, src_1_count,src_2, src_2_count
        )
    