from math import pi
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def radian2Degree(val):
    return val/pi*180

def degree2Radian(val):
    return val/180*pi

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
    # a=tempLoss(output[:, 0:2], target[:, 0:2]) + tempLoss(output[:, 2:4], target[:, 2:4])
    # print("a: ",a)
    # print(output[:, 2:4])
    return torch.min(
        DoALoss(output[:, 0:2], target[:, 0:2]) + DoALoss(output[:, 2:4], target[:, 2:4]),
        DoALoss(output[:, 0:2], target[:, 2:4]) + DoALoss(output[:, 2:4], target[:, 0:2])
    )

if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [0, 0, 0, 3.14],
            [0, 0, 0, 3.14]
        ]
    )

    labels = torch.tensor(
        [
            [0, 3.14, 0, 3.14],
            [0, 3.14, 0, 3.14]
        ]
    )
    print(outputs.shape)
    # loss = nn.MSELoss(outputs, labels)
    loss = cost_multiSound(outputs, labels)
    loss = cost_manhattan(outputs, labels)

    print(loss)