from math import pi
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import os
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

# from nvidia.dali import pipeline_def, Pipeline
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types

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
    hrirSet_l = l['content_m'][0][0]
    hrirSet_r = r['content_m'][0][0]
    elev = l['elev_v'][0][0]
    azim = l['azim_v'][0][0]
    fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

    locLabel = np.hstack((elev, azim)) 
    print("locLabel shape: ", locLabel.shape, " (order: elev, azim)") # locLabel shape: (187, 2)
    # print(locLabel[0:5])

    # 0: left-ear 1: right-ear
    hrirSet = np.vstack((np.reshape(hrirSet_l, (1,) + hrirSet_l.shape),
                            np.reshape(hrirSet_r, (1,) + hrirSet_r.shape)))
    hrirSet = np.transpose(hrirSet, (1,0,2))
    
    return hrirSet, locLabel, fs_HRIR

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

    def __call__(self, idx):
        """
        Args:
            idx (int): the index of the subject
        """
        assert(
            0 <= idx < self.num_subject
        ), f"Invalid subject index. Available indexes: {0, self.num_subject}"

        subject = self.subjects[idx]
        filename = os.path.join(self.names[idx], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
        m = loadmat(filename, struct_as_record=True)

        l, r = m['l_eq_hrir_S'], m['r_eq_hrir_S']
        hrirSet_l = l['content_m'][0][0]
        hrirSet_r = r['content_m'][0][0]
        elev = l['elev_v'][0][0]
        azim = l['azim_v'][0][0]
        fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

        locLabel = np.hstack((elev, azim)) 
        # print("locLabel shape: ", locLabel.shape, " (order: elev, azim)") # locLabel shape: (187, 2)
        # print(locLabel[0:5])

        # 0: left-ear 1: right-ear
        hrirSet = np.vstack((np.reshape(hrirSet_l, (1,) + hrirSet_l.shape),
                                np.reshape(hrirSet_r, (1,) + hrirSet_r.shape)))
        hrirSet = np.transpose(hrirSet, (1,0,2))

        return hrirSet, locLabel, fs_HRIR

class LocRegion:
    def __init__(self, locLabel):
        self.locLabel = locLabel
        self.Nloc = locLabel.shape[0]
        self.high_left = []
        self.high_right = []
        self.low_left = []
        self.low_right = []
        self.elev_dict = {}
        self.azim_dict = {}
        for i in range(self.locLabel.shape[0]):
            elev = self.locLabel[i, 0]
            azim = self.locLabel[i, 1]
            if not(elev in self.elev_dict.keys()):
                self.elev_dict[elev] = [i]
            else:
                self.elev_dict[elev].append(i)
            if not(azim in self.azim_dict.keys()):
                self.azim_dict[azim] = [i]
            else:
                self.azim_dict[azim].append(i)
                
            if self.locLabel[i, 0] > 0 and 0 < self.locLabel[i, 1] < 180:
                self.high_left.append(i)
            elif self.locLabel[i, 0] <= 0 and 0 < self.locLabel[i, 1] < 180:
                self.low_left.append(i)
            elif self.locLabel[i, 0] > 0 and self.locLabel[i, 1] > 180:
                self.high_right.append(i)
            elif self.locLabel[i, 0] <= 0 and self.locLabel[i, 1] > 180:
                self.low_right.append(i)
        print(f"Number of locations in each region: {len(self.high_left)}, {len(self.low_left)}, {len(self.high_right)}, {len(self.low_right)}")
    
    def getLocRegion(self):
        return (self.high_left, self.low_left, self.high_right, self.low_right)

    def whichRegion(self, locIndex):
        if locIndex in self.high_left:
            return "HL"
        elif locIndex in self.low_left:
            return "LL"
        elif locIndex in self.high_right:
            return "HR"
        elif locIndex in self.low_right:
            return "LR"
        else:
            return "None"
    
    def getElev(self, locIndex):
        return self.locLabel[locIndex,0]

    def getAzim(self, locIndex):
        return self.locLabel[locIndex,1]


if __name__ == "__main__":
    # path = "C:/Users/mynam/Desktop/SSSL-desktop/HRTF/IRC*"
    # # path = "./HRTF/IRC*"
    # hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    # print(f"{hrirSet.shape}")

    load_hrir = LoadHRIR(path="C:/Users/mynam/Desktop/SSSL-desktop/HRTF/IRC*")
    hrirSet, locLabel, fs_HRIR = load_hrir(4)
    print(f"{hrirSet.shape}")

    loc_region = LocRegion(locLabel)
    print(f"{loc_region.elev_dict}")
    print(f"{loc_region.azim_dict}")