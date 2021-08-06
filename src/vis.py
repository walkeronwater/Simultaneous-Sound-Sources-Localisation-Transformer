import matplotlib.pyplot as plt
import librosa
import librosa.display
from glob import glob
import os
import torch
import torch.nn as nn
from scipy.fft import fft, ifft

from utils import *
from load_data import *

def plotHistory(checkptPath):
    checkptPath = glob(os.path.join(checkptPath, "curve*"))
    print("Number of epochs: ",len(checkptPath))

    
    history = {
        'train_acc': [],
        'train_loss': [],
        'valid_acc': [],
        'valid_loss': []
    }
    for i in range(len(checkptPath)):
        checkpt = torch.load(checkptPath[i])
        for idx in history.keys():
            history[idx].append(checkpt[idx])
    print(history['train_acc'])

    plt.plot(history['train_acc'])
    plt.plot(history['valid_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.show()

    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    pass