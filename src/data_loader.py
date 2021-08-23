from math import pi
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import os
import shutil
import re
from glob import glob
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from numba import jit, njit
from numba.experimental import jitclass

from utils import *

# from nvidia.dali import pipeline_def, Pipeline
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types

#[TODO]: deprecate
def loadHRIR(path):
    names = []
    names += glob(path)

    splitnames = [os.path.split(name) for name in names]

    p = re.compile('IRC_\d{4,4}')

    subjects = [int(name[4:8]) for base, name in splitnames 
                            if not (p.match(name[-8:]) is None)]

    k = 0
    subject = subjects[k]
    print(subjects)

    for k in range(len(names)):
        subject = subjects[k]
        # filename = os.path.join(names[k], 'IRC_' + str(subject))
        filename = os.path.join(names[k], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
    # print(filename)

    m = loadmat(filename, struct_as_record=True)
    # print(m.keys())
    # print(m['l_eq_hrir_S'].dtype)

    l, r = m['l_eq_hrir_S'], m['r_eq_hrir_S']
    hrir_set_l = l['content_m'][0][0]
    hrir_set_r = r['content_m'][0][0]
    elev = l['elev_v'][0][0]
    azim = l['azim_v'][0][0]
    fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

    loc_label = np.hstack((elev, azim)) 
    print("Loc Label shape: ", loc_label.shape, " (order: elev, azim)") # loc_label shape: (187, 2)
    # print(loc_label[0:5])

    # 0: left-ear 1: right-ear
    hrir_set = np.vstack((np.reshape(hrir_set_l, (1,) + hrir_set_l.shape),
                            np.reshape(hrir_set_r, (1,) + hrir_set_r.shape)))
    hrir_set = np.transpose(hrir_set, (1,0,2))
    
    return hrir_set, loc_label, fs_HRIR

class LoadHRIR:
    def __init__(self, path):
        """
        Args:
            path (str): the HRIR directory that ends with IRC*
        """

        self.names = []
        self.names += glob(path)
        self.num_subject = len(self.names)
        splitnames = [os.path.split(name) for name in self.names]

        p = re.compile('IRC_\d{4,4}')

        self.subjects = [int(name[4:8]) for base, name in splitnames 
                                if not (p.match(name[-8:]) is None)]

        self.load_subject(0)
        print(f"HRIR set shape: {self.hrir_set.shape}, loc Label shape: {self.loc_label.shape}")
        self.loc_region = LocRegion(self.loc_label)

    def load_subject(self, subject_idx):
        """
        Args:
            subject_idx (int): the index of the subject
        """
        assert(
            0 <= subject_idx < self.num_subject
        ), f"Invalid subject index. Available indexes: {0, self.num_subject}"

        subject = self.subjects[subject_idx]
        filename = os.path.join(self.names[subject_idx], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
        m = loadmat(filename, struct_as_record=True)

        l, r = m['l_eq_hrir_S'], m['r_eq_hrir_S']
        hrir_set_l = l['content_m'][0][0]
        hrir_set_r = r['content_m'][0][0]
        elev = l['elev_v'][0][0]
        azim = l['azim_v'][0][0]
        # remove the location at elev 90
        elev = np.delete(elev,(-1), axis = 0)
        azim = np.delete(azim,(-1), axis = 0)

        self.fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

        self.loc_label = np.hstack((elev, azim)) 
        # print("Loc label shape: ", loc_label.shape, " (order: elev, azim)") # loc_label shape: (187, 2)

        # 0: left-ear 1: right-ear
        self.hrir_set = np.vstack((np.reshape(hrir_set_l, (1,) + hrir_set_l.shape),
                                np.reshape(hrir_set_r, (1,) + hrir_set_r.shape)))
        self.hrir_set = np.transpose(self.hrir_set, (1,0,2))

    def __call__(self, loc_idx):
        return self.hrir_set[loc_idx], self.loc_label[loc_idx]

class LocRegion:
    def __init__(self, loc_label):
        self.loc_label = loc_label
        self.Nloc = loc_label.shape[0]
        self.high_left = []
        self.high_right = []
        self.low_left = []
        self.low_right = []
        self.elev_dict = {}
        self.azim_dict = {}
        for i in range(self.loc_label.shape[0]):
            elev = self.loc_label[i, 0]
            azim = self.loc_label[i, 1]
            if not(elev in self.elev_dict.keys()):
                self.elev_dict[elev] = [i]
            else:
                self.elev_dict[elev].append(i)
            if not(azim in self.azim_dict.keys()):
                self.azim_dict[azim] = [i]
            else:
                self.azim_dict[azim].append(i)
                
            if self.loc_label[i, 0] > 0 and 0 < self.loc_label[i, 1] < 180:
                self.high_left.append(i)
            elif self.loc_label[i, 0] < 0 and 0 < self.loc_label[i, 1] < 180:
                self.low_left.append(i)
            elif self.loc_label[i, 0] > 0 and self.loc_label[i, 1] > 180:
                self.high_right.append(i)
            elif self.loc_label[i, 0] < 0 and self.loc_label[i, 1] > 180:
                self.low_right.append(i)
        print(f"Number of locations in each region: {len(self.high_left)}, {len(self.low_left)}, {len(self.high_right)}, {len(self.low_right)}")
    
    def getLocRegion(self):
        return (self.high_left, self.low_left, self.high_right, self.low_right)

    def whichRegion(self, loc_idx):
        if loc_idx in self.high_left:
            return "HL"
        elif loc_idx in self.low_left:
            return "LL"
        elif loc_idx in self.high_right:
            return "HR"
        elif loc_idx in self.low_right:
            return "LR"
        else:
            return "None"
    
    def getElev(self, loc_idx):
        return self.loc_label[loc_idx,0]

    def getAzim(self, loc_idx):
        return self.loc_label[loc_idx,1]
    
    def getDiff(self, loc_1, loc_2):
        return getManhattanDiff(loc_1, loc_2)

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
    
    def __call__(self, seq, loc_idx):
        """
        Convolve the audio signal with resampled HRIR for a location index

        Args:
            seq (ndarray): audio signal
            loc_idx (int): location index of the signal

        Return:
            tuple (ndarray, ndarray): left-ear and right-ear signal sequences after convolution
        """
        assert (
            0 <= loc_idx < self.hrir.shape[0]
        ), "Invalid location index"

        sigL = np.convolve(seq, self.hrir[loc_idx, 0])
        sigR = np.convolve(seq, self.hrir[loc_idx, 1])

        if self.val_SNR >= 100:
            return (sigL, sigR)
        else:
            # print(f"current SNR {self.val_SNR} noise type {self.noise_type}")
            return (sigL + self.noiseGenerator(sigL), sigR + self.noiseGenerator(sigR))
    

    def noiseGenerator(self, seq):
        # if self.noise_type.lower() == "gaussian":
        #     sig_power = 10*np.log10(np.mean(np.power(seq, 2)))
        #     noise_power = np.power(10, (sig_power - self.val_SNR)/10)
        #     noise_sig = np.random.normal(0, np.sqrt(noise_power), seq.shape)
            # print(noise_power)
            # return noise_sig
        if self.noise_type.lower() == "gaussian":
            p_sig = np.var(seq)
            p_noise = p_sig/np.power(10, self.val_SNR/10)
            return np.random.normal(0, np.sqrt(p_noise), seq.shape)

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
        freq_axis, time_axis, Sxx = signal.spectrogram(
            seq, self.fs_audio, nfft=1023, mode="magnitude", window=('hamming'),
            nperseg=1023, noverlap=512
        )
        _, _, Phxx = signal.spectrogram(
            seq, self.fs_audio, nfft=1023, mode="phase", window=('hamming'),
            nperseg=1023, noverlap=512
        )

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

    def annotate(self, loc_idx_list: list):
        if self.fileCount == 0:
            if os.path.isfile(self.savePath+'dataLabels.csv'):
                print("Directory exists -- overwriting")
                # if input('Delete saved_cues? ') == 'y':
                #     print('ok')
                shutil.rmtree(self.savePath)
                os.mkdir(self.savePath)
            with open(self.savePath+'dataLabels.csv', 'w') as csvFile:
                csvFile.write(str(self.fileCount))
                for i in loc_idx_list:
                    csvFile.write(',')
                    csvFile.write(str(locIndex2Label(self.locLabel, i, "allClass")))
                csvFile.write('\n')
        else:
            with open(self.savePath+'dataLabels.csv', 'a') as csvFile:
                csvFile.write(str(self.fileCount))
                for i in loc_idx_list:
                    csvFile.write(',')
                    csvFile.write(str(locIndex2Label(self.locLabel, i, "allClass")))
                csvFile.write('\n')

    def __call__(self, cuesList: list, loc_idx_list: list):
        cues = self.concatCues(cuesList)
        self.annotate(loc_idx_list=loc_idx_list)
        torch.save(cues, self.savePath+str(self.fileCount)+'.pt')
        self.fileCount += 1


if __name__ == "__main__":
    # path = "C:/Users/mynam/Desktop/SSSL-desktop/HRTF/IRC*"
    # # path = "./HRTF/IRC*"
    # hrir_set, loc_label, fs_HRIR = loadHRIR(path)
    # print(f"{hrir_set.shape}")

    # load_hrir = LoadHRIR(path="C:/Users/mynam/Desktop/SSSL-desktop/HRTF/IRC*")
    load_hrir = LoadHRIR(path="./HRTF/IRC*")
    hrir_pair, loc_label = load_hrir(loc_idx=0)
    print(f"{hrir_pair.shape, loc_label.shape}")

    # temp = {}
    # count = 1
    # for i in load_hrir.loc_region.high_left + load_hrir.loc_region.high_right + load_hrir.loc_region.low_left + load_hrir.loc_region.low_right:
    #     if not (i in temp.keys()):
    #         temp[i] = 1
    #     count += 1
    # print(count)

    print(load_hrir.loc_region.whichRegion(1))