import soundfile as sf
from scipy import signal
import random
from math import pi
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
from torchsummary import summary

from load_data import *
from utils import *

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
    def __init__(self, filePath, task, Nsound, locLabel, isDebug=False):
        super(MyDataset, self).__init__()
        self.filePath = filePath
        self.task = task
        self.Nsound = Nsound
        self.annotation = pd.read_csv(filePath+"dataLabels.csv", header=None)
        self.isDebug = isDebug
        self.locLabel = locLabel

    def __len__(self):
        return int(self.annotation.iloc[-1, 0] + 1)
    
    def __getitem__(self, pathIndex):
        data = torch.load(self.filePath+str(pathIndex)+".pt")
        if self.Nsound == 1:
            if self.task.lower() == "allclass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 1]))
            elif self.task.lower() == "elevclass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 2]))
            elif self.task.lower() == "azimclass":
                labels = torch.tensor(int(self.annotation.iloc[pathIndex, 3]))
            elif self.task.lower() == "elevregression":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 4], dtype=torch.float32)
            elif self.task.lower() == "azimregression":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 5], dtype=torch.float32)
            elif self.task.lower() == "allregression":
                labels = torch.stack(
                    [
                        torch.tensor(self.annotation.iloc[pathIndex, 4], dtype=torch.float32),
                        torch.tensor(self.annotation.iloc[pathIndex, 5], dtype=torch.float32)
                    ]
                )
            elif self.task.lower() == "multisound":
                labels = torch.tensor(self.annotation.iloc[pathIndex, 3:5].values, dtype=torch.float32)
        else:
            # labels = torch.tensor(self.annotation.iloc[pathIndex, 1:5].values, dtype=torch.float32)
            # labels = torch.stack(
            #         [
            #             torch.tensor(self.annotation.iloc[pathIndex, 1], dtype=torch.float32),
            #             torch.tensor(self.annotation.iloc[pathIndex, 2], dtype=torch.float32),
            #             torch.tensor(self.annotation.iloc[pathIndex, 3], dtype=torch.float32),
            #             torch.tensor(self.annotation.iloc[pathIndex, 4], dtype=torch.float32)
            #         ]
                # )
            if self.task.lower() == "allregression":
                label = []
                for i in self.annotation.iloc[pathIndex].values[1:]:
                    label.extend(self.locLabel[i])
                # print(label)
                labels = degree2Radian(torch.tensor(label, dtype=torch.float32))
                # labels = torch.tensor(self.annotation.iloc[pathIndex].values[1:], dtype=torch.float32)
            elif self.task.lower() == "allclass":
                labels = torch.tensor(self.annotation.iloc[pathIndex].values[1:], dtype=torch.int)

        if self.isDebug:
            print("pathIndex: ", pathIndex)
            print("label:", labels)

        return data, labels


class CuesDataset(torch.utils.data.Dataset):
    def __init__(self, filePath, task, Nsound, locLabel, coordinates="spherical", isDebug=False):
        super(CuesDataset, self).__init__()
        self.filePath = filePath
        self.task = task
        # self.Nsound = Nsound
        self.annotation = pd.read_csv(filePath+"dataLabels.csv", header=None)
        self.coordinates = coordinates
        self.isDebug = isDebug
        self.locLabel = locLabel
        
        self.Nsound = self.annotation.iloc[0].values.shape[0] - 1
        print(f"self.Nsound: {self.Nsound}")
        assert(
            self.Nsound == Nsound
        ), "Number of sound sources doesn't match."
        self.Nfreq, self.Ntime, self.Ncues = list(torch.load(filePath+"/0.pt").shape)

    def __len__(self):
        return int(self.annotation.iloc[-1, 0] + 1)
    
    def __getitem__(self, pathIndex):
        data = torch.load(self.filePath+str(pathIndex)+".pt")

        from_csv = self.annotation.iloc[pathIndex].values
        
        if self.coordinates.lower() == "spherical":
            labels = torch.empty((2*(from_csv.shape[0]-1)))
        
            # read starting from the second element in the pathIndex row
            for i in range(1, from_csv.shape[0]):
                labels[2*(i-1):2*(i-1)+2] = degree2Radian(torch.from_numpy(self.locLabel[from_csv[i]]))

        elif self.coordinates.lower() == "cartesian":
            labels = torch.empty((3*(from_csv.shape[0]-1)))
        
            # read starting from the second element in the pathIndex row
            for i in range(1, from_csv.shape[0]):
                labels[3*(i-1):3*(i-1)+3] = torch.from_numpy(spherical2Cartesian(self.locLabel[from_csv[i]]))

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

"""
class ModelSelection:
    def __init__(
        self,
        task,
        Nsound
    ):
        self.task = task
        self.Nsound = Nsound

    def __call__(self):
        if self.Nsound = 2:
            if self.task.lower() == "allregression":
                model =
        return
"""

def predNeuron(task):
    if task == "elevClass":
        return 10
    elif task == "azimClass":
        return 24
    elif task == "allClass":
        return 187
    elif task == "elevRegression" or task == "azimRegression":
        return 1
    elif task == "allRegression":
        return 2

def saveParam(epoch, model, optimizer, scheduler, savePath, task):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'task': task
    }, savePath)

def saveCurves(epoch, tl, ta, vl, va, savePath, task):
    torch.save({
        'epoch': epoch,
        'train_loss': tl,
        'train_acc': ta,
        'valid_loss': vl,
        'valid_acc': va,
        'task': task
    }, savePath)

def loadCheckpoint(model, optimizer, scheduler, loadPath, task, phase, whichBest=None):
    if whichBest == "None":
        checkpoint = torch.load(loadPath+"param.pth.tar")
    else:
        checkpoint = torch.load(loadPath+"param"+"_"+whichBest+".pth.tar")

    if checkpoint['task'] == task:
        epoch = checkpoint['epoch']
        print("Model is retrieved at epoch ", epoch)
        # try:
        model.load_state_dict(checkpoint['model'], strict=False)
        # except:
        #     model.load_state_dict(checkpoint['state_dict'])

        trainHistory = glob(os.path.join(loadPath, "curve*"))

        history = {
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'valid_loss': []
        }
        for i in range(len(trainHistory)):
            checkpt = torch.load(
                loadPath+"curve_epoch_"+str(i+1)+".pth.tar"
            )
            for idx in history.keys():
                history[idx].append(checkpt[idx])

        val_loss_optim = history['valid_loss'][epoch-1]
        print("val_loss_optim: ", val_loss_optim)
        print("Corresponding validation accuracy: ",
            history['valid_acc'][epoch-1]
        )

        if phase == "train":
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print("scheduler not found")
            
            preTrainEpoch = len(trainHistory)
            print("Training will start from epoch", preTrainEpoch+1)
            return model, optimizer, scheduler, preTrainEpoch, val_loss_optim
        elif phase == "test":
            return model, val_loss_optim
    else:
        raise SystemExit("Task doesn't match")

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.stop = False
        self.count = 0
        self.val_loss_optim = float('inf')

    def __call__(self, val_loss):
        if self.val_loss_optim < val_loss:
            self.count += 1
        else:
            self.val_loss_optim = val_loss
            self.count = 0

        if self.count >= self.patience:
            self.stop = True

def getLR(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def setLR(newLR, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = newLR
    return optimizer

class SourcePrediction:
    """
    Store the prediction for a single sound source
    """
    def __init__(self):
        self.elev_pred = []
        self.elev_target = []
        self.azim_pred = []
        self.azim_target = []
        self.loc_loss = np.zeros((187, 187), dtype=float)
            
    def __call__(self, outputs, labels, src_loc, src_loss):
        """
        Args:
            outputs: shape of (batch size, 2)
            labels: shape of (batch size, 2)
            src_loc: the location label of a sound source
            src_loss: the loss corresponding to that location
        """
        
        self.elev_pred.extend([radian2Degree(self.unwrapPrediction(i, "elev")) for i in outputs[:, 0].tolist()])
        self.elev_target.extend([radian2Degree(self.unwrapPrediction(i, "elev")) for i in labels[:, 0].tolist()])
        self.azim_pred.extend([radian2Degree(self.unwrapPrediction(i, "azim")) for i in outputs[:, 1].tolist()])
        self.azim_target.extend([radian2Degree(self.unwrapPrediction(i, "azim")) for i in labels[:, 1].tolist()])
        if len(src_loc) == 2:
            self.loc_loss[src_loc[0], src_loc[1]] = src_loss
        elif len(src_loc) == 1:
            self.loc_loss[src_loc] = src_loss
        

        # for i in range(outputs.shape[0]):
        #     self.elev_pred.append(outputs[i, 0].item())
        #     self.elev_target.append(labels[i, 0].item())
        #     self.azim_pred.append(outputs[i, 1].item())
        #     self.azim_target.append(labels[i, 1].item())
    
    def unwrapPrediction(self, val, type_input: str):
        if type_input.lower() == "azim":
            while val > pi:
                val -= 2*pi
            while val < -pi:
                val += 2*pi
        elif type_input.lower() == "elev":
            while val > 2*pi:
                val -= 2*pi
            while val < 0:
                val += 2*pi
            if pi/2 < val < pi*3/2:
                val = pi - val
            elif pi*3/2 < val < pi*2:
                val -= 2*pi
        return val

class VisualisePrediction:
    """
    Support multiple sound-source prediction
    """
    def __init__(self, Nsound):
        self.sound_list = []
        for _ in range(Nsound):
            self.sound_list.append(SourcePrediction())
        self.Nsound = Nsound
        

    def __call__(self, outputs, labels, src_loc: list, src_loss: list):
        """
        Args:
            outputs: shape of (batch size, 2*Nsound)
            labels: shape of (batch size, 2*Nsound)
            loc_pair (list): location index of each sound source
            src_loss (list): loss of each sound source
        """
        for src_idx in range(self.Nsound):
            # print(f"src_loss: {src_loss[src_idx]}, {type(src_loss[src_idx])}")
            self.sound_list[src_idx](outputs[:,2*src_idx:2*(src_idx+1)], labels[:,2*src_idx:2*(src_idx+1)], src_loc, src_loss[src_idx])

    def report(self, src_loc:list, path=None):
        # print(f"{len(self.elev_pred)}, {len(self.elev_target)}, {len(self.azim_pred)}, {len(self.azim_pred)}")
        for src_idx in range(self.Nsound):
            # plt.figure()
            # plt.scatter(
            #     range(187),
            #     np.true_divide(self.sound_list[src_idx].loc_loss.sum(axis=self.Nsound -1-src_idx),(self.sound_list[src_idx].loc_loss!=0).sum(axis=self.Nsound -1-src_idx))
            # )
            # plt.xticks(range(0, 186, 6))
            # # plt.plot(range(187), np.max(self.sound_list[src_idx].loc_loss, axis=self.Nsound -1-src_idx))
            # plt.xlabel("Location")
            # plt.ylabel("Max angle error in degree")
            # plt.title(f"Loss of sound source {src_idx}")
            # plt.grid()
            # plt.show()

            # x = np.linspace(-45, 90, 100)
            # y = x
            # plt.figure()
            # plt.scatter(self.sound_list[src_idx].elev_target, self.sound_list[src_idx].elev_pred, color='blue')
            # plt.plot(x, y,'-r')
            # plt.xticks(range(-45, 91, 15))
            # plt.yticks(range(-45, 91, 15))
            # plt.xlabel("Ground truth")
            # plt.ylabel("Prediction")
            # plt.title(f"Elevation of sound source {src_idx}")
            # plt.grid()
            # plt.show()

            # x = np.linspace(0, 345, 100)
            # y = x
            # plt.figure()
            # plt.scatter(self.sound_list[src_idx].azim_target, self.sound_list[src_idx].azim_pred, color='blue')
            # plt.plot(x, y,'-r')
            # plt.xticks(range(0, 360, 15))
            # plt.yticks(range(0, 360, 15))
            # plt.xlabel("Ground truth")
            # plt.ylabel("Prediction")
            # plt.title(f"Azimuth of sound source {src_idx}")
            # plt.grid()
            # plt.show()

            plt.figure(figsize=(12, 6), dpi=100)
            plt.scatter(self.sound_list[src_idx].azim_target, self.sound_list[src_idx].elev_target, color='red')
            plt.scatter(self.sound_list[src_idx].azim_pred, self.sound_list[src_idx].elev_pred, color='blue')
            for j in range(len(self.sound_list[src_idx].azim_target)):
                plt.plot(
                    [self.sound_list[src_idx].azim_target[j], self.sound_list[src_idx].azim_pred[j]],
                    [self.sound_list[src_idx].elev_target[j], self.sound_list[src_idx].elev_pred[j]],
                    color="black", linewidth=0.5, linestyle="--"
                )
            plt.xticks(np.linspace(-180, 180, 25))
            plt.yticks(np.linspace(-90, 90, 13))
            # plt.axis([-180, 180, -90, 90])
            # plt.axis("square", emit=False)
            plt.xlabel("Azimuth")
            plt.ylabel("Elevation")
            fig_title = [str(i) for i in src_loc]
            loc_pair_name = ' and '.join(fig_title)
            plt.title(f"Prediction and target of sound source {src_idx} for location {loc_pair_name}")
            plt.grid()
            if not path:
                plt.show()
            else:
                plt.savefig(f"{path}loc_{loc_pair_name}_src_{src_idx}.png")

    def connectPoints(self):
        pass


    def plotRegression(self, pred_list, target_list):
        pass


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


    train_dataset = CuesDataset(
        filePath="./saved_0808_temp/train/",
        task="allclass",
        Nsound=2,
        locLabel=locLabel,
        coordinates="cartesian"
    )

    train_loader = MultiEpochsDataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        persistent_workers=False
    )

    labels = torch.tensor(
        [
            [1, 4],
            [1, 5],
        ]
    )

    count=0
    for i, (inputs, labels) in enumerate(train_loader):
        count+=1
        print(labels)
        raise SystemExit


    print(labels.size(0))
    # labels = labels.unsqueeze(0)
    target = torch.zeros(labels.size(0), 187).scatter_(1, labels, 1.)
    print(target.shape)

    outputs = torch.rand((32, 187))
    # outputs = torch.nn.functional.one_hot(outputs.to(torch.int64), num_classes=187)
    # outputs = torch.argmax(outputs, dim=1)
    print(outputs.shape)
    criterion = torch.nn.BCEWithLogitsLoss()
    for i, (inputs, labels) in enumerate(train_loader):
        print(labels)
        labels_hot = torch.zeros(labels.size(0), 187).scatter_(1, labels.to(torch.int64), 1.).float()

        print("label_hot shape", labels_hot.shape)

        _, hit = torch.topk(labels_hot, k=2, dim=1)
        hit, _ = torch.sort(hit, dim=1, descending=False)
        print("convert back", hit)

        loss = criterion(outputs.float(), labels_hot)
        print(loss)
        raise SystemExit('dbg')
