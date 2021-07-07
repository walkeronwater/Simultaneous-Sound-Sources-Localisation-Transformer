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

if __name__ == "__main__":
    lenSliceInSec = 0.5   # length of audio slice in sec
    fileCount = 0   # count the number of data samples
    Nfreq = 512
    Ntime = 45
    Ncues = 5
    Nloc = 24
    Nsample = Nloc * 2000
    if "cues_" in globals() or "cues_" in locals():
        del cues_
    if "labels_" in globals() or "labels_" in locals():
        del labels_
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    labels_ = torch.zeros((Nsample, 1))
    valSNRList = [-20,-15,-10,-5,0,5,10,15,20,100]
    for audioIndex in range(len(path)):
        if fileCount == Nsample:
            break
        print("Audio index: ", audioIndex)
        audio, fs_audio = sf.read(path[audioIndex])
        audio = librosa.resample(audio, fs_audio, fs_HRIR)

        audioSliceList = audioSliceGenerator(audio, fs_HRIR, lenSliceInSec)

        for sliceIndex in range(len(audioSliceList)):
            if fileCount == Nsample:
                break
            audioSlice = audio[audioSliceList[sliceIndex]]

            for locIndex in range(Nloc):
                if fileCount == Nsample:
                    break
                sigLeft = np.convolve(audioSlice, hrirSet[24*3+locIndex, 0])
                sigRight = np.convolve(audioSlice, hrirSet[24*3+locIndex, 1])

                # print("Location index: ", locIndex)
                # showSpectrogram(sigLeft, fs_HRIR)
                # showSpectrogram(sigRight, fs_HRIR)
                
                specLeft = calSpectrogram(sigLeft)
                specRight = calSpectrogram(sigRight)
                
                for valSNR in valSNRList:
                    if fileCount == Nsample:
                        break
                        # raise SystemExit("Cues_ tensor is full.")
                    # specLeft_noise = calSpectrogram(sigLeft + noiseGenerator(sigLeft, valSNR))
                    # specRight_noise = calSpectrogram(sigRight + noiseGenerator(sigRight, valSNR))

                    ipdCues = calIPD(specLeft, specRight)
                    # ipdCues_noise = calIPD(specLeft_noise,specRight_noise)
                    # showSpectrogram(ipdCues, fs_HRIR, "IPD without noise", isLog=False)
                    # showSpectrogram(ipdCues_noise, fs_HRIR, "IPD with noise SNR "+str(valSNR), isLog=False)
                    
                    ildCues = calILD(specLeft, specRight)
                    # ildCues_noise = calILD(specLeft_noise,specRight_noise)
                    # showSpectrogram(ildCues, fs_HRIR, "ILD without noise", isLog=False)
                    # showSpectrogram(ildCues_noise, fs_HRIR, "ILD with noise SNR "+str(valSNR), isLog=False)

                    r_l, theta_l  = cartesian2euler(specLeft)
                    r_r, theta_r  = cartesian2euler(specLeft)
                    # print(ipdCues.shape)
                    # print(r_l.shape, theta_l.shape, r_r.shape, theta_r.shape)
                    # raise SystemExit("Debugging")

                    cues = torch.from_numpy(np.concatenate((np.expand_dims(ipdCues, axis=-1), np.expand_dims(ildCues, axis=-1)),axis=-1))
                    # IPD, r_l, theta_l, r_r, theta_r
                    cues = torch.from_numpy(
                        np.concatenate((np.expand_dims(ipdCues, axis=-1),
                                        np.expand_dims(r_l, axis=-1),
                                        np.expand_dims(theta_l, axis=-1),
                                        np.expand_dims(r_r, axis=-1),
                                        np.expand_dims(theta_r, axis=-1),
                                        ),axis=-1)
                    )
                    labels = torch.tensor([locIndex])

                    cues_[fileCount] = cues
                    labels_[fileCount] = labels

                    # if fileCount == 0 :
                    #     cues_ = cues.unsqueeze(0)
                    #     labels_ = labels.unsqueeze(0)
                    # else:
                    #     cues_ = torch.cat([cues_, cues.unsqueeze_(0)])
                    #     labels_ = torch.cat([labels_, labels.unsqueeze_(0)])

                    fileCount += 1
                    if fileCount % (Nloc*len(valSNRList)) == 0:
                        print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                            fileCount // (Nloc*len(valSNRList)))
            
    print("fileCount: ", fileCount)
    print(cues[-1])