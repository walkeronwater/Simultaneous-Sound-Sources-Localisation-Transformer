import torch
from torch import Tensor
import torch.nn.functional as F

def DoALoss(output, target):
    sine_term = torch.sin(output[:, 0]) * torch.sin(target[:, 0])
    cosine_term = torch.cos(output[:, 0]) * torch.cos(target[:, 0]) * torch.cos(target[:, 1] - output[:, 1])
    loss = torch.acos(F.hardtanh(sine_term + cosine_term, min_val=-1, max_val=1))

    return loss

if __name__ == "__main__":
    a = torch.randn((32,2))
    b = torch.randn((32,2))
    loss = DoALoss(a, b)

    print(loss)