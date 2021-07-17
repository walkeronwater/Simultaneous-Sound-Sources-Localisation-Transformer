import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def DoALoss(output, target):
    # target should be (elev, azim)
    # sine: azimuth
    sine_term = torch.sin(output[:, 1]) * torch.sin(target[:, 1])
    cosine_term = torch.cos(output[:, 1]) * torch.cos(target[:, 1]) * torch.cos(target[:, 0] - output[:, 0])
    loss = torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    return torch.mean(loss)

if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [0.78,0],
            [-0.78,0]
        ]
    )

    labels = torch.tensor(
        [
            [-0.78,0],
            [-0.78,0]
            
        ]
    )
    print(outputs.shape)
    # loss = nn.MSELoss(outputs, labels)
    loss = DoALoss(outputs, labels)

    print(loss)