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

# from nvidia.dali import pipeline_def, Pipeline
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types

def loadHRIR(path):
    names = []
    names += glob(path)
    print(names[0])

    splitnames = [os.path.split(name) for name in names]
    print(len(splitnames))

    p = re.compile('IRC_\d{4,4}')
    print(p)

    subjects = [int(name[4:8]) for base, name in splitnames 
                            if not (p.match(name[-8:]) is None)]
    print(subjects)

    k = 0
    subject = subjects[k]

    for k in range(len(names)):
        subject = subjects[k]
        # filename = os.path.join(names[k], 'IRC_' + str(subject))
        filename = os.path.join(names[k], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
    #     print(filename)

    m = loadmat(filename, struct_as_record=True)
    print(m.keys())
    print(m['l_eq_hrir_S'].dtype)

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
    print("hrirSet shape: ", hrirSet.shape)
    
    return hrirSet, locLabel, fs_HRIR

class LocRegion:
    def __init__(self, locLabel):
        self.locLabel = locLabel
        
        self.high_left = []
        self.high_right = []
        self.low_left = []
        self.low_right = []
        for i in range(self.locLabel.shape[0]):
            if self.locLabel[i, 0] > 0 and 0 < self.locLabel[i, 1] < 180:
                self.high_left.append(i)
            elif self.locLabel[i, 0] < 0 and 0 < self.locLabel[i, 1] < 180:
                self.low_left.append(i)
            elif self.locLabel[i, 0] > 0 and 0 < self.locLabel[i, 1] > 180:
                self.high_right.append(i)
            elif self.locLabel[i, 0] < 0 and 0 < self.locLabel[i, 1] > 180:
                self.low_right.append(i)
    
    def getLocRegion(self):
        return (self.high_left, self.low_left, self.high_right, self.low_right)

    def whichRegion(self, locIndex):
        if locIndex in self.high_left:
            return "HL"
        elif locIndex in self.low_left:
            return "LL"
        elif locIndex in self.high_right:
            return "HR"
        elif locIndex in self.low_left:
            return "LR"
        else:
            return "None"

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filePath, task, Nsound, isDebug=False):
        super(MyDataset, self).__init__()
        self.filePath = filePath
        self.task = task
        self.Nsound = Nsound
        self.annotation = pd.read_csv(filePath+"dataLabels.csv", header=None)
        self.isDebug = isDebug

    def __len__(self):
        return int(self.annotation.iloc[-1, 0] + 1)
    
    def __getitem__(self, pathIndex):
        if self.Nsound == 1:
            data = torch.load(self.filePath+str(pathIndex)+".pt")
            if self.task == "allClass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 1]))
            elif self.task == "elevClass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 2]))
            elif self.task == "azimClass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 3]))
            elif self.task == "elevRegression":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 4], dtype=torch.float32)
            elif self.task == "azimRegression":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 5], dtype=torch.float32)
            elif self.task == "allRegression":
                labels = torch.stack([
                    torch.tensor(self.annotation.iloc[pathIndex, 4], dtype=torch.float32),
                    torch.tensor(self.annotation.iloc[pathIndex, 5], dtype=torch.float32)]
                )
            elif self.task.lower() == "multisound":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 1:3].values, dtype=torch.float32)
        else:
            data = torch.load(self.filePath+str(pathIndex)+".pt")
            labels = torch.tensor(self.annotation.iloc[pathIndex, 1:5].values, dtype=torch.float32)

        if self.isDebug:
            print("pathIndex: ", pathIndex)
            print("label:", labels)

        return data, labels

def splitDataset(batchSize, trainValidSplit: list, numWorker, dataset):
    Ntrain = round(trainValidSplit[0]*dataset.__len__())
    if Ntrain % batchSize == 1:
        Ntrain -=1
    Nvalid = round(trainValidSplit[1]*dataset.__len__())
    if Nvalid % batchSize == 1:
        Nvalid -=1
    # Ntest = dataset.__len__() - Ntrain - Nvalid
    # if Ntest % batchSize == 1:
    #     Ntest -=1
    print("Dataset separation: ", Ntrain, Nvalid)

    train, valid = torch.utils.data.random_split(dataset, [Ntrain, Nvalid], generator=torch.Generator().manual_seed(24))
    # train_loader = DataLoader(dataset=train, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)
    # valid_loader = DataLoader(dataset=valid, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)

    train_loader = MultiEpochsDataLoader(dataset=train, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)
    valid_loader = MultiEpochsDataLoader(dataset=valid, batch_size=batchSize, shuffle=True, num_workers=numWorker, persistent_workers=False)

    return train_loader, valid_loader

# batch_size = 8
# class SimplePipeline(Pipeline):
#     def __init__(self, filePath, batch_size, num_threads, device_id):
#         super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
#         self.input = ops.FileReader(file_root = filePath)
#         # instead of path to file directory file with pairs image_name image_label_value can be provided
#         # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
#         self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

#     def define_graph(self):
#         jpegs, labels = self.input()
#         images = self.decode(jpegs)
#         return (images, labels)

# @pipeline_def
# def caffe_pipeline(num_gpus):
#     device_id = Pipeline.current().device_id
#     jpegs, labels = fn.readers.caffe(
#         name='Reader', path=lmdb_folder, random_shuffle=True, shard_id=device_id, num_shards=num_gpus)
#     images = fn.decoders.image(jpegs, device='mixed')
#     images = fn.resize(
#         images,
#         resize_shorter=fn.random.uniform(range=(256, 480)),
#         interp_type=types.INTERP_LINEAR)
#     images = fn.crop_mirror_normalize(
#             images,
#             crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
#             crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
#             dtype=types.FLOAT,
#             crop=(227, 227),
#             mean=[128., 128., 128.],
#             std=[1., 1., 1.])

#     return images, labels



if __name__ == "__main__":
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)

    # dataset = MyDataset("./saved_cues_temp/", "azimClass", isDebug=True)
    # train_loader, valid_loader = splitDataset(32, [0.8, 0.2], 0, dataset)

    # for i, data in enumerate(train_loader):
    #     inputs, labels = data
    # for i in range(25):
    #     print(dataset[i][1].shape)

    loc_region = LocRegion(locLabel)
    
    high_left, low_left, high_right, low_right = loc_region.getLocRegion()

    print(loc_region.whichRegion(123))