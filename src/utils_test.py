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

from data_loader import *
from utils_train import *
from utils import *

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

    def report(self, fixed_src:list, path=None):
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
            plt.legend(["target", "prediction"])
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
            # plt.legend(*scatter.legend_elements())
            plt.xlabel("Azimuth")
            plt.ylabel("Elevation")

            elev, azim = fixed_src[0], fixed_src[1]
            plt.title(f"Prediction and target of sound source {src_idx} when source 0 is fixed at elevation {elev} and azim {azim}")
            plt.grid()
            if not path:
                plt.show()
            else:
                plt.savefig(f"{path}fixed_src0_elev{elev}_azim{azim}_src_{src_idx}.png")
            plt.close()
            
    def connectPoints(self):
        pass


    def plotRegression(self, pred_list, target_list):
        pass


class Confusion:
    """
    Implement LR UD FB confusion
    """
    def __init__(self):
        """
        squared error of UD, LR FB confusion
        """
        self.se_UD = 0.0
        self.se_LR = 0.0
        self.se_FB = 0.0
    
    def __call__(self, outputs, labels):
        for i in range(outputs.shape[0]):
            while outputs[i, 1] > pi*2:
                outputs[i, 1] -= pi*2
            while outputs[i, 1] < 0:
                outputs[i, 1] += pi*2
            while outputs[i, 0] > pi*2:
                outputs[i, 0] -= pi*2
            while outputs[i, 0] < 0:
                outputs[i, 0] += pi*2
            if pi/2 < outputs[i, 0] < pi*3/2:
                outputs[i, 0] = pi - outputs[i, 0]
            elif pi*3/2 < outputs[i, 0] < pi*2:
                outputs[i, 0] -= 2*pi

        self.up_down(outputs, labels)
        self.left_right(outputs, labels)
        self.front_back(outputs, labels)

    def LR_loss(self, output):
        angle_diff = torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.cos(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )
        for i in range(angle_diff.shape[0]):
            if pi < output[i, 1] < pi*2:
                angle_diff[i] = -angle_diff[i]
                # print("LR: ",radian2Degree(angle_diff[i]))
        return angle_diff
    
    def FB_loss(self, output):
        angle_diff = torch.acos(
            F.hardtanh(
                torch.sqrt(
                    torch.square(torch.cos(output[:, 0])) * torch.square(torch.sin(output[:, 1]))
                    + torch.square(torch.sin(output[:, 0]))
                ), min_val=-1, max_val=1
            )
        )
        for i in range(angle_diff.shape[0]):
            if pi/2 <= output[i, 1] <= pi*3/2:
                angle_diff[i] = -angle_diff[i]
                # print("FB: ",radian2Degree(angle_diff[i]))
        return angle_diff
    
    def up_down(self, pred, target):
        self.se_UD += torch.sum(torch.square(pred[:,0] - target[:,0])).item()

    def left_right(self, pred, target):
        self.se_LR += torch.sum(torch.square(self.LR_loss(pred) - self.LR_loss(target))).item()
    
    def front_back(self, pred, target):
        self.se_FB += torch.sum(torch.square(self.FB_loss(pred) - self.FB_loss(target))).item()

class TwoSourceError:
    def __init__(self):
        self.elev_pred_1 = []
        self.elev_target_1 = []
        self.azim_pred_1 = []
        self.azim_target_1 = []
        self.elev_pred_2 = []
        self.elev_target_2 = []
        self.azim_pred_2 = []
        self.azim_target_2 = []
        self.loss_1 = []
        self.loss_2 = []
        self.loss_dict = {}

    def __call__(self, outputs, labels, loss_1, loss_2):
        self.elev_pred_1.extend(
            [self.unwrapPrediction(i, "elev") for i in outputs[:,0].tolist()]
        )
        self.elev_target_1.extend(labels[:,0].tolist())
        self.azim_pred_1.extend(
            [self.unwrapPrediction(i, "azim") for i in outputs[:,1].tolist()]
        )
        self.azim_target_1.extend(labels[:,1].tolist())
        self.elev_pred_2.extend(
            [self.unwrapPrediction(i, "elev") for i in outputs[:,2].tolist()]
        )
        self.elev_target_2.extend(labels[:,2].tolist())
        self.azim_pred_2.extend(
            [self.unwrapPrediction(i, "azim") for i in outputs[:,3].tolist()]
        )
        self.azim_target_2.extend(labels[:,3].tolist())
        self.loss_1.extend(loss_1.tolist())
        self.loss_2.extend(loss_2.tolist())
        self.loss_dict

    def unwrapPrediction(self, val, type_input:str):
        if type_input.lower() == "azim":
            while val > 2*pi:
                val -= 2*pi
            while val < 0:
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

    def plotPrediction(self):
        # marker_size = [i for i in self.loss_1]
        # marker_size = [1 for i in self.loss_1]
        # marker_size = [5*round(i//15+1) for i in self.loss_1]
        marker_size = []
        for i in self.loss_1:
            if i<=15:
                marker_size.append(5)
            elif i<=30:
                marker_size.append(2.5)
            elif i<=45:
                marker_size.append(1)
            elif i<=60:
                marker_size.append(0.5)
            else:
                marker_size.append(0)

        plt.figure(figsize=(12, 6), dpi=100)
        plt.scatter(
            [radian2Degree(i) for i in self.azim_target_1],
            [radian2Degree(i) for i in self.elev_target_1],
            color='red',
            s = marker_size
        )
        plt.scatter(
            [radian2Degree(i) for i in self.azim_pred_1],
            [radian2Degree(i) for i in self.elev_pred_1],
            color='blue',
            s = [5 for i in self.loss_1]
        )
        plt.xticks(np.linspace(0, 345, 24))
        plt.yticks(np.linspace(-90, 75, 12))
        plt.xlabel("Azimuth")
        plt.ylabel("Elevation")
        plt.legend(["target", "prediction"])
        plt.title("Source 1")
        plt.grid()
        plt.show()
        plt.close()

        # marker_size = [i for i in self.loss_2]
        # marker_size = [1 for i in self.loss_2]
        # marker_size = [5*round(i//15+1) for i in self.loss_2]
        marker_size = []
        for i in self.loss_2:
            if i<=15:
                marker_size.append(5)
            elif i<=30:
                marker_size.append(2.5)
            elif i<=45:
                marker_size.append(1)
            elif i<=60:
                marker_size.append(0.5)
            else:
                marker_size.append(0)

        plt.figure(figsize=(12, 6), dpi=100)
        plt.scatter(
            [radian2Degree(i) for i in self.azim_target_2],
            [radian2Degree(i) for i in self.elev_target_2],
            color='red',
            s = marker_size
        )
        plt.scatter(
            [radian2Degree(i) for i in self.azim_pred_2],
            [radian2Degree(i) for i in self.elev_pred_2],
            color='blue',
            s = [5 for i in self.loss_2]
        )
        plt.xticks(np.linspace(0, 345, 24))
        plt.yticks(np.linspace(-90, 75, 12))
        plt.xlabel("Azimuth")
        plt.ylabel("Elevation")
        plt.legend(["target", "prediction"])
        plt.title("Source 2")
        plt.grid()
        plt.show()
        plt.close()

    def plotError(self):
        plt.figure(figsize=(12, 6), dpi=100)
        plt.scatter(
            [radian2Degree(i) for i in self.azim_target_1],
            self.loss_1
        )
        plt.scatter(
            [radian2Degree(i) for i in self.azim_target_2],
            self.loss_2
        )
        plt.xticks(np.linspace(0, 345, 24))
        plt.yticks(np.linspace(0, 345, 24))
        plt.xlabel("Azimuth")
        plt.ylabel("RMS angle error")
        plt.legend(["Source 1", "Source 2"])
        plt.title("Azimuth error")
        plt.grid()
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 6), dpi=100)
        plt.scatter(
            [radian2Degree(i) for i in self.elev_target_1],
            self.loss_1
        )
        plt.scatter(
            [radian2Degree(i) for i in self.elev_target_2],
            self.loss_2
        )
        plt.xticks(np.linspace(-90, 75, 12))
        plt.yticks(np.linspace(-90, 75, 12))
        plt.xlabel("Elevation")
        plt.ylabel("RMS angle error")
        plt.legend(["Source 1", "Source 2"])
        plt.title("Elevation error")
        plt.grid()
        plt.show()
        plt.close()