from math import pi
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from utils import radian2Degree, degree2Radian

class CostFunc:
    def __init__(
        self,
        task,
        Nsound,
        device,
        coordinates="spherical"
    ):
        self.task = task
        self.Nsound = Nsound
        self.device = device
        self.coordinates = coordinates
        if Nsound == 1 and ("class" in self.task.lower()):
            self.cls_criterion = nn.CrossEntropyLoss()
        elif Nsound == 2 and ("class" in self.task.lower()):
            self.cls_criterion = nn.BCEWithLogitsLoss()

    def __call__(self, outputs, labels):
        if self.Nsound == 1:
            if "regression" in self.task.lower():
                return torch.sqrt(torch.mean(torch.square(self.calDoALoss(outputs, labels))))
            elif "class" in self.task.lower():
                return self.cls_criterion(outputs, labels)
        else:
            if "regression" in self.task.lower():
                if self.coordinates.lower() == "spherical":
                    return torch.sqrt(torch.mean(torch.square(self.calDoALoss(outputs[:, 0:2], labels[:, 0:2]) + self.calDoALoss(outputs[:, 2:4], labels[:, 2:4]))))
                elif self.coordinates.lower() == "cartesian":
                    return torch.sqrt(torch.mean(torch.square(self.calDoALossCartesian(outputs[:, 0:3], labels[:, 0:3]) + self.calDoALossCartesian(outputs[:, 3:6], labels[:, 3:6]))))
            elif "class" in self.task.lower():
                labels_hot = torch.zeros(labels.size(0), 187).to(self.device)
                labels_hot = labels_hot.scatter_(1, labels.to(torch.int64), 1.).float()
                return self.cls_criterion(outputs.float(), labels_hot)

    def calDoALoss(self, outputs, labels):
        """
        labels should be (elev, azim)
        sine term: azimuth
        """
        sine_term = torch.sin(outputs[:, 0]) * torch.sin(labels[:, 0])
        cosine_term = torch.cos(outputs[:, 0]) * torch.cos(labels[:, 0]) * torch.cos(labels[:, 1] - outputs[:, 1])
        loss = torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))
        return torch.absolute(loss)

    def calDoALossCartesian(self, outputs, labels):
        """
        labels should be (x, y, z)
        """
        dot_product = torch.sum(torch.mul(outputs, labels), dim=1)
        cross_product = torch.cross(outputs, labels, dim=1)
        loss = torch.atan(F.hardtanh(torch.norm(cross_product, dim=1)/dot_product, min_val=-1, max_val=1))
        return torch.absolute(loss)

class Confusion:
    def __init__(self):
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
                print("LR: ",radian2Degree(angle_diff[i]))
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

def DoALoss(output, target):
    # target should be (elev, azim)
    # sine: azimuth
    sine_term = torch.sin(output[:, 0]) * torch.sin(target[:, 0])
    cosine_term = torch.cos(output[:, 0]) * torch.cos(target[:, 0]) * torch.cos(target[:, 1] - output[:, 1])
    loss = torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    return torch.absolute(loss)

def cost_manhattan(output, target):
    # torch.stack(torch.sin(target[:, 0]) torch.cos(target[:, 0]))

    return torch.sqrt(torch.square(output[:, 0] - target[:, 0]) + torch.square(output[:, 2] - target[:, 2])) + torch.sqrt(torch.square(output[:, 1] - target[:, 1]) + torch.square(output[:, 3] - target[:, 3]))

def cost_multiSound(output, target):
    # output and target should be of size (batch size, 4)
    # return torch.min(
    #     DoALoss(output[:, 0:2], target[:, 0:2]) + DoALoss(output[:, 2:4], target[:, 2:4]),
    #     DoALoss(output[:, 0:2], target[:, 2:4]) + DoALoss(output[:, 2:4], target[:, 0:2])
    # )
    return DoALoss(output[:, 0:2], target[:, 0:2]) + DoALoss(output[:, 2:4], target[:, 2:4])

if __name__ == "__main__":
    cost_func = CostFunc(
        task="allRegression",
        Nsound=2,
        device="cpu",
        coordinates="cartesian"
    )

    outputs_reg = torch.tensor(
        [
            [0, -pi/2]
        ]
    )
    labels_reg = torch.tensor(
        [
            [0, pi/2]
        ]
    )

    outputs_cls = torch.rand((2,187))
    labels_cls = torch.tensor(
        [
            [1,2],
            [0,2]
        ]
    )

    # print(cost_func(outputs_reg, labels_reg))
    confusion = Confusion()
    confusion(outputs_reg, labels_reg)
    print(f"{confusion.se_LR}, {confusion.se_UD}, {confusion.se_FB}")