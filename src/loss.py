from math import pi
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def radian2degree(val):
    return val/pi*180

def DoALoss(output, target):
    # target should be (elev, azim)
    # sine: azimuth
    sine_term = torch.sin(output[:, 0]) * torch.sin(target[:, 0])
    cosine_term = torch.cos(output[:, 0]) * torch.cos(target[:, 0]) * torch.cos(target[:, 1] - output[:, 1])
    loss = torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    return torch.mean(loss)

if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [1.047, 0],
            [0.7854, 0]
        ]
    )

    labels = torch.tensor(
        [
            [-1.57, 3.14],
            [-1.57, 3.14]
        ]
    )
    print(outputs.shape)
    # loss = nn.MSELoss(outputs, labels)
    loss = DoALoss(outputs, labels)

    print(radian2degree(loss.item()))