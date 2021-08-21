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
from numba import jit, njit
from numba.experimental import jitclass
from skimage.restoration import unwrap_phase
# from pydub import AudioSegment


# [TODO] method to log IPD cues and spectral cues
'''def cuesLog():'''

def locIndex2Label(locLabel, locIndex, task):
    if task.lower() == "elevclass":
        # range of elevation: -45 to 90 degrees
        labels = int(((locLabel[locIndex, 0]+45) % 150)/15)
    elif task.lower()  == "azimclass":
        # range of elevation: 0 to 345 degrees
        labels = int((locLabel[locIndex, 1] % 360)/15)
    elif task.lower()  == "allclass":
        labels = int(locIndex)
    elif task.lower()  == "elevregression":
        labels = locLabel[locIndex, 0]/180.0*pi
    elif task.lower()  == "azimregression":
        labels = locLabel[locIndex, 1]/180.0*pi
    elif task.lower()  == "allregression":
        labels = torch.tensor(
            [
                locLabel[locIndex, 0]/180.0*pi,
                locLabel[locIndex, 1]/180.0*pi
            ], dtype=torch.float32
        )
    return labels

def radian2Degree(val):
    return val/pi*180

def degree2Radian(val):
    return val/180*pi

def linear2dbfs(val, factor="amplitude"):
    if factor.lower() == "amplitude":
        return 20*np.log10(val)
    elif factor.lower() == "power":
        return 10*np.log10(val**2)

def spherical2Cartesian(val):
    """
    val (nd array): order (elev, azimuth)
    """
    elev = val[0]
    azim = val[1]
    elev = degree2Radian(elev)
    azim = degree2Radian(azim)  
    x = np.cos(azim) * np.cos(elev)
    y = np.sin(azim) * np.cos(elev)
    z = np.sin(elev)

    return np.array([x,y,z])

def cartesian2Spherical(val):
    """
    val (nd array): order (x, y, z)
    """
    # x, y, z = val[0], val[1], val[2]
    # r = np.sqrt(x**2 + y**2 + z**2)
    # elev = np.arcsin(z/r)
    # azim = np.arctan(y/x)
    # return np.array((elev, azim))

    batch_size, tensor_len = val.shape
    out = torch.empty(batch_size, int(tensor_len/3*2))
    for i in range(0, val.shape[-1], 3):
        r = torch.sqrt(val[:,i] **2 + val[:,i+1] **2 + val[:,i+2] **2)
        out[:,int(2*i/3)] = torch.asin(torch.div(val[:,i+2], r))
        temp = torch.atan2(val[:,i+1], val[:,i])
        temp[temp < 0] += 2*pi
        out[:,int(2*i/3+1)] = temp

    # for i in range(0, val.shape[-1], 3):
    #     elev = torch.arcsin(val[:,i+2]])
    #     azim = torch.arcsin(torch.div(val[:,i+1], val[:,i]))

    # return torch.stack([elev,azim], dim=-1)
    return out

def getAngleDiff(loc_label_1, loc_label_2):
    """
    Args:
        loc_label_1: nd array
        loc_label_1: nd array
    Returns:
        angle_diff: nd array - dtype float32
    """
    sine_term = np.sin(loc_label_1[0]) * np.sin(loc_label_2[0])
    cosine_term = np.cos(loc_label_1[0]) * np.cos(loc_label_2[0]) * np.cos(loc_label_1[1] - loc_label_2[1])
    angle_diff = np.absolute(
                    np.arccos(
                        np.clip(
                            sine_term + cosine_term,
                            -1, 1
                        )
                    ), dtype=np.float32
                )
    print(angle_diff.dtype)
    return angle_diff

def getManhattanDiff(loc_label_1, loc_label_2):
    """
    Args:
        loc_label_1: nd array
        loc_label_1: nd array
    Returns:
        angle_diff: nd array - dtype float32
    """
    return np.sum(np.absolute(loc_label_1 - loc_label_2), dtype=np.float32)

if __name__ == "__main__":
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    
    speech_male_path = glob(os.path.join("./audio_train/speech_male/*"))
    speech_female_path = glob(os.path.join("./audio_train/speech_female/*"))

    speech_male = AudioSignal(path=speech_male_path[0], slice_duration=1)
    speech_female = AudioSignal(path=speech_female_path[0], slice_duration=1)
    sig_sliced = speech_male(idx=1)
    print(f"length of slice list: {len(sig_sliced)}")

    binaural_sig = BinauralSignal(hrirSet, fs_HRIR, speech_male.fs_audio)
    sigL, sigR = binaural_sig(sig_sliced, loc_idx=160)
    sigL_, sigR_ = binaural_sig(sig_sliced, loc_idx=105)
    sigL_2, sigR_2 = binaural_sig(speech_female(idx=1), loc_idx=160)

    binaural_cues = BinauralCues(speech_male.fs_audio, "minmax")
    
    magL, phaseL, magR, phaseR = binaural_cues(sigL, sigR)
    print(f"magL shape: {magL.shape}")
    magL_, phaseL_, magR_, phaseR_ = binaural_cues(sigL_, sigR_)
    magL_2, phaseL_2, magR_2, phaseR_2 = binaural_cues(sigL_2, sigR_2)

    vis_cues = VisualiseCues(speech_male.fs_audio, binaural_cues.freq_axis, binaural_cues.time_axis)
    # vis_cues.showBinauralSig(sig_sliced, sigL, sigR)
    # vis_cues.showSpectrogram(magL, figTitle="left-ear mag", isLog=True)
    # vis_cues.showSpectrogram(magL_, figTitle="left-ear mag", isLog=True)
    # vis_cues.showSpectrogram(phaseL, figTitle="left-ear phase", isLog=False)
    # vis_cues.showSpectrogram(phaseL_, figTitle="right-ear phase", isLog=False)
    vis_cues.showCues(magL-magL_, figTitle="left-ear mag")
    vis_cues.showCues(magL_-magL_2, figTitle="left-ear mag")
    vis_cues.showCues(phaseL-phaseL_, figTitle="left-ear phase")
    vis_cues.showCues(phaseL_-phaseL_2, figTitle="left-ear phase")

    raise SystemExit("debugging")

    # trainAudioPath = glob(os.path.join("./audio_train/*"))

    # _, fs_audio = sf.read(trainAudioPath[0])
    # temp = librosa.resample(hrirSet[0, 0], fs_HRIR, fs_audio)
    # hrirSet_re = np.empty(hrirSet.shape[0:2]+temp.shape)
    # for i in range(hrirSet.shape[0]):
    #     hrirSet_re[i, 0] = librosa.resample(hrirSet[i, 0], fs_HRIR, fs_audio)
    #     hrirSet_re[i, 1] = librosa.resample(hrirSet[i, 1], fs_HRIR, fs_audio)
    # print(hrirSet_re.shape)
    # del temp, hrirSet

    # cuesShape = CuesShape()
    # lenSliceInSec = 0.5
    # normalise = Preprocess(prep_method="normalise")
    # standardise = Preprocess(prep_method="standardise")

    # audio_1, fs_audio_1 = sf.read(trainAudioPath[0])
    # audio_2, fs_audio_2 = sf.read(trainAudioPath[1])
    # audioSliceList_1 = audioSliceGenerator(audio_1, fs_HRIR, lenSliceInSec)
    # audioSliceList_2 = audioSliceGenerator(audio_2, fs_HRIR, lenSliceInSec)

    # sliceIndex_1 = 0
    # sliceIndex_2 = 1

    # audioSlice_1 = audio_1[audioSliceList_1[sliceIndex_1]]
    # audioSlice_2 = audio_2[audioSliceList_2[sliceIndex_2]]
    # locIndex_1 = 5
    # locIndex_2 = 39

    # sigLeft_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 0])
    # sigRight_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 1])
    # sigLeft_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 0])
    # sigRight_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 1])

    # binaural_cues = BinauralCues(prep_method="normalise", fs_audio=fs_audio)

    # ipd, magL, phaseL, magR, phaseR = binaural_cues(sigLeft_1+sigLeft_2, sigRight_1+sigRight_2)

    # vis_cues = VisualiseCues(fs_audio=fs_audio, Nfreq=binaural_cues.Nfreq, Ntime=binaural_cues.Ntime)
    
    # vis_cues(ipd, figTitle="IPD")
    # vis_cues(phaseL, figTitle="phaseL")
    # vis_cues(phaseR, figTitle="phaseR")

    # save_cues = SaveCues(savePath="./saved_0208_temp/", locLabel=locLabel)
    # for _ in range(10):
    #     save_cues([ipd, magL, phaseL, magR, phaseR], [locIndex_1, locIndex_2])


    # trainAudioPath = glob(os.path.join("./audio_train/speech_male/*"))
    # print(trainAudioPath)
    # audio_1, fs_audio_1 = sf.read(trainAudioPath[4])

    # slices, powers = audioSliceGenerator(audio_1, fs_HRIR, 0.5, threshold=0, isDebug=True)
    # print(len(slices))
    # print(powers)
    vis = VisualiseCues(fs_audio=16000, Nfreq=512, Ntime=44)
    # binaural_cues = BinauralCues(prep_method="normalise", fs_audio=fs_audio_1)
    # spec = binaural_cues.calSpectrogram(audio_1, fs_audio_1)
    # print(np.mean(np.absolute(spec)))
    # vis.showSpectrogram(spec, fs_audio_1, figTitle="spectrogram", isLog=True)

    temp = torch.load("./saved_0308_temp/train/2.pt")
    print(temp.shape)
    vis.showSpectrogram(temp[:,:,0].numpy(), 16000, figTitle="ipd", isLog=False)