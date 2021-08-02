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

class BinauralCues:
    def __init__(self, fs_audio, prep_method):
        self.preprocess = Preprocess(prep_method=prep_method)
        self.fs_audio = fs_audio
        # self.sigL = sigL
        # self.sigR = sigR
        self.flag = False
        self.Nfreq = None
        self.Ntime = None
    
    def calSpectrogram(self, seq, fs):
        Nfft = 1023
        Zxx = librosa.stft(seq, n_fft=Nfft, hop_length=512)
        Nfreq = librosa.fft_frequencies(sr=fs, n_fft=Nfft)
        length = seq.shape[0] / fs
        Ntime = np.linspace(0., length, seq.shape[0])
        Ntime = librosa.frames_to_time(range(0, Zxx.shape[1]), sr=fs, hop_length=512, n_fft=Nfft)
        if not self.flag:
            self.Nfreq = Nfreq
            self.Ntime = Ntime
            self.flag = True
        return Nfreq, Ntime, Zxx

    def calIPD(self, specL, specR):
        ipd = np.angle(np.divide(specL, specR, out=np.zeros_like(specL), where=np.absolute(specR)!=0))
        ipd = unwrap_phase(ipd)
        # ipd = np.unwrap(ipd)
        
        # get ITD:
        # if not self.Nfreq_vis:
        #     self.Nfreq_vis = np.tile(self.Nfreq, (512,1))
        #     self.Nfreq_vis = np.transpose(self.Nfreq_vis, axes=None)
        #     self.Nfreq_vis = np.flip(self.Nfreq_vis)
        # itd = ipd * Nfreq_vis * 1/(2*pi)
        return ipd

    def calILD(self, specL, specR):
        ild = 20*np.log10(np.divide(np.absolute(specL), np.absolute(specR), out=np.zeros_like(np.absolute(specL)), where=np.absolute(specR)!=0))
        return ild

    def cartesian2euler(self, spec):
        # x = spec.real
        # y = spec.imag
        # mag = np.sqrt(x**2+y**2)
        # theta = np.angle(np.divide(y, x, where=x!=0))
        mag = np.abs(spec)
        phase = np.angle(spec)
        phase = unwrap_phase(phase)
        # phase = np.unwrap(phase)
        return mag, phase

    def __call__(self, sigL, sigR):
        Nfreq, Ntime, specL = self.calSpectrogram(sigL, self.fs_audio)
        _, _, specR = self.calSpectrogram(sigR, self.fs_audio)
        ipd = self.calIPD(specL, specR)
        magL, phaseL = self.cartesian2euler(specL)
        magR, phaseR = self.cartesian2euler(specR)

        ipd, magL, phaseL, magR, phaseR = (self.preprocess(i) for i in [ipd, magL, phaseL, magR, phaseR])
        
        return (ipd, magL, phaseL, magR, phaseR)

class VisualiseCues:
    def __init__(self, fs_audio, Nfreq, Ntime):
        self.fs_audio = fs_audio
        self.Nfreq = Nfreq
        self.Ntime = Ntime
    
    def showSpectrogram(self, Zxx, fs, figTitle, isLog=True):
        fig, ax = plt.subplots()
        print("Spectrogram shape: ", Zxx.shape)

        if isLog:
            img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(Zxx),ref=np.max),
                                        sr=fs, hop_length=512, fmax=fs/2,
                                        y_axis='linear', x_axis='time', ax=ax)
        else:
            img = librosa.display.specshow(np.abs(Zxx),
                                        sr=fs, hop_length=512, fmax=fs/2,
                                        y_axis='linear', x_axis='time', ax=ax)
        ax.set_title(figTitle)
        if isLog:
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            fig.colorbar(img, ax=ax, format="%+2.0f")
        # fig.set_figheight(5)
        # fig.set_figwidth(5)
        plt.show()
    
    def showCues(self, data, Nfreq, Ntime, figTitle):
        # data shape: (Nfreq, Ntime)
        for i in range(0, self.Ntime.size, 10):
            plt.plot(Nfreq, data[:,i])
        plt.xlabel("Frequency")
        plt.ylabel(figTitle)
        plt.title(figTitle)
        plt.grid()
        plt.show()

    def __call__(self, data, figTitle):
        self.showSpectrogram(data, fs=self.fs_audio, figTitle=figTitle, isLog=False)
        self.showCues(data, Nfreq=self.Nfreq, Ntime=self.Ntime, figTitle=figTitle)

    
# [TODO] method to log IPD cues and spectral cues
'''def cuesLog():'''


class SaveCues:
    def __init__(self, savePath, locLabel):
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        self.savePath = savePath
        self.fileCount = 0
        self.locLabel = locLabel

    def concatCues(self, cuesList: list):
        cuesShape = cuesList[0].shape
        lastDim = len(cuesList)
        cues = torch.zeros(cuesShape+(lastDim,), dtype=torch.float)

        for i in range(lastDim):
            cues[:,:,i] = torch.from_numpy(cuesList[i])

        return cues

    def annotate(self, locIndex: list):
        if self.fileCount == 0:
            if os.path.isfile(self.savePath+'dataLabels.csv'):
                print("Directory exists -- overwriting")
                # if input('Delete saved_cues? ') == 'y':
                #     print('ok')
                shutil.rmtree(self.savePath)
                os.mkdir(self.savePath)
            with open(self.savePath+'dataLabels.csv', 'w') as csvFile:
                csvFile.write(str(self.fileCount))
                for i in locIndex:
                    csvFile.write(',')
                    csvFile.write(str(locIndex2Label(self.locLabel, i, "allClass")))
                csvFile.write('\n')
        else:
            with open(self.savePath+'dataLabels.csv', 'a') as csvFile:
                csvFile.write(str(self.fileCount))
                for i in locIndex:
                    csvFile.write(',')
                    csvFile.write(str(locIndex2Label(self.locLabel, i, "allClass")))
                csvFile.write('\n')

    def __call__(self, data, locIndex):
        cues = self.concatCues(data)
        self.annotate(locIndex=locIndex)
        torch.save(cues, self.savePath+str(self.fileCount)+'.pt')
        self.fileCount += 1


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

def radian2Degree(val):
    return val/pi*180

def degree2Radian(val):
    return val/180*pi

if __name__ == "__main__":
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

    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    print(hrirSet.shape)

    trainAudioPath = glob(os.path.join("./audio_train/*"))

    _, fs_audio = sf.read(trainAudioPath[0])
    temp = librosa.resample(hrirSet[0, 0], fs_HRIR, fs_audio)
    hrirSet_re = np.empty(hrirSet.shape[0:2]+temp.shape)
    for i in range(hrirSet.shape[0]):
        hrirSet_re[i, 0] = librosa.resample(hrirSet[i, 0], fs_HRIR, fs_audio)
        hrirSet_re[i, 1] = librosa.resample(hrirSet[i, 1], fs_HRIR, fs_audio)
    print(hrirSet_re.shape)
    del temp, hrirSet

    cuesShape = CuesShape()
    lenSliceInSec = 0.5
    normalise = Preprocess(prep_method="normalise")
    standardise = Preprocess(prep_method="standardise")

    audio_1, fs_audio_1 = sf.read(trainAudioPath[0])
    audio_2, fs_audio_2 = sf.read(trainAudioPath[1])
    audioSliceList_1 = audioSliceGenerator(audio_1, fs_HRIR, lenSliceInSec)
    audioSliceList_2 = audioSliceGenerator(audio_2, fs_HRIR, lenSliceInSec)

    sliceIndex_1 = 0
    sliceIndex_2 = 1

    audioSlice_1 = audio_1[audioSliceList_1[sliceIndex_1]]
    audioSlice_2 = audio_2[audioSliceList_2[sliceIndex_2]]
    locIndex_1 = 5
    locIndex_2 = 39

    sigLeft_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 0])
    sigRight_1 = np.convolve(audioSlice_1, hrirSet_re[locIndex_1, 1])
    sigLeft_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 0])
    sigRight_2 = np.convolve(audioSlice_2, hrirSet_re[locIndex_2, 1])

    binaural_cues = BinauralCues(prep_method="normalise", fs_audio=fs_audio)

    ipd, magL, phaseL, magR, phaseR = binaural_cues(sigLeft_1+sigLeft_2, sigRight_1+sigRight_2)

    vis_cues = VisualiseCues(fs_audio=fs_audio, Nfreq=binaural_cues.Nfreq, Ntime=binaural_cues.Ntime)
    
    vis_cues(ipd, figTitle="IPD")
    vis_cues(phaseL, figTitle="phaseL")
    vis_cues(phaseR, figTitle="phaseR")

    save_cues = SaveCues(savePath="./saved_0208_temp/", locLabel=locLabel)
    for _ in range(10):
        save_cues([ipd, magL, phaseL, magR, phaseR], [locIndex_1, locIndex_2])