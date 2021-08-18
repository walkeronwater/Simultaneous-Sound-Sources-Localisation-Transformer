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

from load_data import *


# method to normalise a sequence which can be broadcasted to a sequence of sequence
# min-max/standardise/L2 norm for each tensor like an image
class Preprocess:
    def __init__(self, prep_method: str):
        """
        Args:
            prep_method (str): the preprocessing method
        """
        self.prep_method = prep_method
        if self.prep_method.lower() == "standardise":
            print("Preprocessing method: standardise")
        elif self.prep_method.lower() == "normalise":
            print("Preprocessing method: normalise")
        elif self.prep_method.lower() == "minmax":
            print("Preprocessing method: minmax")
        else:
            print("Preprocessing method: none")
        # [TODO]: bandpass the input signal for testing

    def __call__(self, seq):
        """
        Args:
            seq (ndarray): left-ear or right-ear signal
        
        Returns:
            seq (ndarray): preprocessed signal
        """
        if self.prep_method.lower() == "standardise":
            return (seq - np.mean(seq))/(np.std(seq))
        elif self.prep_method.lower() == "normalise":
            return seq/np.linalg.norm(seq, ord=1)
        elif self.prep_method.lower() == "minmax":
            return (seq - np.min(seq))/(np.max(seq) - np.min(seq))
        else:
            return seq

class AudioSignal:
    def __init__(self, path, slice_duration):
        """
        read an audio file, calculate the mean power in dBFS

        Args:
            path (str): directory of audio files.
            slice_duration (float): duration of audio slices in second 
        """
        self.sig, self.fs_audio = sf.read(path)
        self.slice_duration = slice_duration
        # mean power in dbfs
        self.mean_power = 10*np.log10(np.mean(np.power(self.sig, 2)))
        self.slice_list = self.audioSliceGenerator(threshold=self.mean_power)
    def __call__(self, idx):
        """
        Args:
            idx (int): the index of sliced signal

        Return:
            sig (list): sliced signal
        """
        assert(
            0 <= idx < len(self.slice_list)
        ), print(f"Invalid slice index, valid range: {0, len(self.slice_list)}")
        
        return self.sig[self.slice_list[idx]]

    def audioSliceGenerator(self, threshold=0.01):
        """
        prepare slices of audio based on the mean power threshold

        Return:
            slice_list (list): a list of slice index ranges
        """
        slice_len = round(self.fs_audio * self.slice_duration)

        slice_list = []
        # threshold for spectrum power
        for i in range(self.sig.size//slice_len):
            sliced = self.sig[slice_len*i : slice_len*(i+1)]
            if 10*np.log10(np.mean(np.power(sliced, 2))) > threshold:
                slice_list.append(range(slice_len*i, slice_len*(i+1)))
        return slice_list

    def apply_gain(self, sliced_sig, target_power):
        mean_power = 10*np.log10(np.mean(np.power(sliced_sig, 2)))
        sliced_sig *= np.power(10, (target_power - self.mean_power)/20)
        return sliced_sig

class BinauralSignal:
    def __init__(self, hrir, fs_hrir, fs_audio, val_SNR=100, noise_type="Gaussian"):
        """
        resample HRIR to the sampling frequency of audio files

        Args:
            hrir (ndarray): hrir sets with shape (187, 2, 512).
            fs_hrir (int): sampling frequency of HRIRs.
            fs_audio (int): sampling frequency of audio files.
            val_SNR (int): noise SNR if <100
            noise_type (str): type of corrupted noise
        """
        self.val_SNR = val_SNR
        self.noise_type = noise_type

        # resampling HRIRs
        temp = librosa.resample(hrir[0,0], fs_hrir, fs_audio)
        hrir_re = np.empty(hrir.shape[0:2]+temp.shape)
        for i in range(hrir.shape[0]):
            hrir_re[i, 0] = librosa.resample(hrir[i, 0], fs_hrir, fs_audio)
            hrir_re[i, 1] = librosa.resample(hrir[i, 1], fs_hrir, fs_audio)
        print(f"HRIR shape after resampling: {hrir_re.shape}")
        self.hrir = hrir_re
    
    def __call__(self, seq, locIndex):
        """
        Convolve the audio signal with resampled HRIR for a location index

        Args:
            seq (ndarray): audio signal
            locIndex (int): location index of the signal

        Return:
            tuple (ndarray, ndarray): left-ear and right-ear signal sequences after convolution
        """
        assert (
            0 <= locIndex < self.hrir.shape[0]
        ), "Invalid location index"

        sigL = np.convolve(seq, self.hrir[locIndex, 0])
        sigR = np.convolve(seq, self.hrir[locIndex, 1])

        if self.val_SNR >= 100:
            return (sigL, sigR)
        else:
            # print(f"current SNR {self.val_SNR} noise tyep {self.noise_type}")
            return (sigL + self.noiseGenerator(sigL), sigR + self.noiseGenerator(sigR))
    

    def noiseGenerator(self, seq):
        if self.noise_type.lower() == "gaussian":
            sig_power = 10*np.log10(np.mean(np.power(seq, 2)))
            noise_power = np.power(10, (sig_power - self.val_SNR)/10)
            noise_sig = np.random.normal(0, np.sqrt(noise_power), seq.shape)
            return noise_sig

class BinauralCues:
    def __init__(self, fs_audio, prep_method):
        """
        Args:
            fs_audio (int): sampling frequency of audio files
            prep_method (str): the preprocessing method
        """
        self.preprocess = Preprocess(prep_method=prep_method)
        self.fs_audio = fs_audio
        # self.sigL = sigL
        # self.sigR = sigR
        self.flag = False
        self.freq_axis = None
        self.time_axis = None
        self.Nfreq = None
        self.Ntime = None
    
    def calSpectrogram(self, seq):
        """
        Args:
            seq (ndarray): left-ear or right-ear signal
        Return:
            Sxx (ndarray): magnitude spectrogram
            Phxx (ndarray): phase spectrogram
        """
        Nfft = 1023

        # using Librosa packages
        # Zxx = librosa.stft(seq, n_fft=Nfft, hop_length=512)
        # if not self.flag:
        #     freq_axis = librosa.fft_frequencies(sr=self.fs_audio, n_fft=Nfft)
        #     length = seq.shape[0] / self.fs_audio
        #     time_axis = np.linspace(0., length, seq.shape[0])
        #     time_axis = librosa.frames_to_time(range(0, Zxx.shape[1]), sr=self.fs_audio, hop_length=512, n_fft=Nfft)
        #     self.freq_axis = freq_axis
        #     self.time_axis = time_axis
        #     self.Nfreq = freq_axis.size
        #     self.Ntime = time_axis.size
        #     self.flag = True
        # preprocess the STFT spectrum
        # self.preprocess(Zxx)
        # return Zxx

        # using scipy packages
        freq_axis, time_axis, Sxx = signal.spectrogram(seq, self.fs_audio, nfft=1023, mode="magnitude")
        _, _, Phxx = signal.spectrogram(seq, self.fs_audio, nfft=1023, mode="phase")

        if not self.flag:
            self.freq_axis = freq_axis
            self.time_axis = time_axis
            self.Nfreq = freq_axis.size
            self.Ntime = time_axis.size
            self.flag = True

        return Sxx, Phxx

    def calIPD(self, specL, specR):
        ipd = np.angle(np.divide(specL, specR, out=np.zeros_like(specL), where=np.absolute(specR)!=0))
        # ipd = unwrap_phase(ipd)
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
        # phase = unwrap_phase(phase)
        # phase = np.unwrap(phase)
        return mag, phase

    def __call__(self, sigL, sigR):
        """
        Calculate all the cues

        Args:
            sigL, sigR: left-ear and right-ear signal
        Returns:
            IPD, ILD, MagL, PhaseL, MagR, PhaseR
        """
        # specL = self.calSpectrogram(sigL)
        # specR = self.calSpectrogram(sigR)
        # ipd = self.calIPD(specL, specR)
        # ild = self.calILD(specL, specR)
        magL, phaseL = self.calSpectrogram(sigL)
        magR, phaseR = self.calSpectrogram(sigR)
        # print(f"min values of mag data {np.min(magL)}, {np.min(magR)}\n \
        #         {np.max(magL)}{np.max(magR)}")

        magL, magR = (self.preprocess(i) for i in [20*np.log10(magL), 20*np.log10(magR)])
        phaseL, phaseR = (self.preprocess(i) for i in [phaseL, phaseR])
        
        return (magL, phaseL, magR, phaseR)

class VisualiseCues:
    def __init__(self, fs_audio, freq_axis, time_axis):
        """
        Args:
            fs_audio (int): sampling frequency of audio files
            freq_axis (ndarray): 1-d frequency axis from 0 to fs/2
            time_axis (ndarray): 1-d time axis
        """
        self.fs_audio = fs_audio
        self.freq_axis = freq_axis
        self.time_axis = time_axis
        self.Nfreq = freq_axis.size
        self.Ntime = time_axis.size

    def showBinauralSig(self, audio_sig, sigL, sigR):
        plt.plot(audio_sig)
        plt.plot(sigL)
        plt.plot(sigR)
        plt.xlabel("Sample")
        plt.ylabel("Magnitude")
        plt.legend(["audio","left-ear","right-ear"])
        plt.title("Visualisation in time-domain")
        plt.grid()
        plt.show()
        
    def showSpectrogram(self, Zxx, figTitle, isLog=True):
        fig, ax = plt.subplots()
        print("Spectrogram shape: ", Zxx.shape)

        Zxx_ = librosa.amplitude_to_db(np.abs(Zxx)) if isLog else Zxx
        
        img = librosa.display.specshow(
            Zxx_,
            sr=self.fs_audio, hop_length=512, fmax=self.fs_audio/2,
            y_axis='linear', x_axis='time', ax=ax
        )

        ax.set_title(figTitle)
        if isLog:
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            fig.colorbar(img, ax=ax, format="%+2.0f")
        # fig.set_figheight(5)
        # fig.set_figwidth(5)
        plt.show()
    
    def showCues(self, data, figTitle):
        # data shape: (Nfreq, Ntime)
        for i in range(0, self.Ntime, 10):
            plt.plot(self.freq_axis, data[:,i])
        plt.xlabel("Frequency")
        plt.ylabel(figTitle)
        plt.title(figTitle)
        plt.legend(range(0, self.Ntime, 10))
        plt.grid()
        plt.show()

    # def __call__(self, data, figTitle):
    #     self.showSpectrogram(data, figTitle=figTitle, isLog=False)
    #     self.showCues(data, figTitle=figTitle)

# [TODO] method to log IPD cues and spectral cues
'''def cuesLog():'''

class SaveCues:
    def __init__(self, savePath, locLabel):
        """
        Args:
            fs_audio (int): sampling frequency of audio files
            locLabel
        """
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        self.savePath = savePath
        self.fileCount = 0
        self.locLabel = locLabel

    def concatCues(self, cuesList: list):
        cuesShape = cuesList[0].shape
        Ncues = len(cuesList)
        cues = torch.zeros(cuesShape+(Ncues,), dtype=torch.float)

        for i in range(Ncues):
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

    def __call__(self, cuesList: list, locIndex: list):
        cues = self.concatCues(cuesList)
        self.annotate(locIndex=locIndex)
        torch.save(cues, self.savePath+str(self.fileCount)+'.pt')
        self.fileCount += 1

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
    if factor.lower == "amplitude":
        return 20*np.log10(val)
    elif factor.lower == "power":
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

if __name__ == "__main__":
    # class CuesShape:
    #     def __init__(
    #         self,
    #         Nfreq = 512,
    #         Ntime = 44,
    #         Ncues = 5,
    #         Nloc = 187,
    #         lenSliceInSec = 0.5,
    #         valSNRList = [-5,0,5,10,15,20,25,30,35]
    #     ):
    #         self.Nfreq = Nfreq
    #         self.Ntime = Ntime
    #         self.Ncues = Ncues
    #         self.Nloc = Nloc
    #         self.lenSliceInSec = lenSliceInSec
    #         self.valSNRList = valSNRList
    val = np.array(
        [0.433, 0.75, 0.5],
        [0.433, 0.75, 0.5]
    )
    print(cartesian2Spherical(val))
    
    raise SystemExit

    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    print(hrirSet.shape)
    for i in range(187):
        x,y,z = spherical2Cartesian(locLabel[i,0], locLabel[i,1])
        print(locLabel[i],x,y,z)
    raise SystemExit
    speech_male_path = glob(os.path.join("./audio_train/speech_male/*"))
    speech_female_path = glob(os.path.join("./audio_train/speech_female/*"))

    speech_male = AudioSignal(path=speech_male_path[0], slice_duration=1)
    sig_sliced = speech_male(idx=1)
    print(f"length of slice list: {len(sig_sliced)}")

    locIndex = 155
    binaural_sig = BinauralSignal(hrirSet, fs_HRIR, speech_male.fs_audio)
    sigL, sigR = binaural_sig(sig_sliced, locIndex)

    binaural_cues = BinauralCues(speech_male.fs_audio, "normalise")
    binaural_cues_noprep = BinauralCues(speech_male.fs_audio, "standardise")
    
    magL, phaseL, magR, phaseR = binaural_cues(sigL, sigR)
    magL_, phaseL_, magR_, phaseR_ = binaural_cues_noprep(sigL, sigR)

    vis_cues = VisualiseCues(speech_male.fs_audio, binaural_cues.freq_axis, binaural_cues.time_axis)
    # vis_cues.showBinauralSig(sig_sliced, sigL, sigR)
    # vis_cues.showSpectrogram(magL, figTitle="left-ear mag", isLog=True)
    # vis_cues.showSpectrogram(magL_, figTitle="left-ear mag", isLog=True)
    # vis_cues.showSpectrogram(phaseL, figTitle="left-ear phase", isLog=False)
    # vis_cues.showSpectrogram(phaseL_, figTitle="right-ear phase", isLog=False)
    vis_cues.showCues(magL, figTitle="left-ear mag")
    vis_cues.showCues(magL_, figTitle="left-ear mag")
    vis_cues.showCues(phaseL, figTitle="left-ear phase")
    vis_cues.showCues(phaseL_, figTitle="left-ear phase")

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