import matplotlib.pyplot as plt
import librosa
import librosa.display
from glob import glob
import os
import torch
import torch.nn as nn

# visualise the spectrogram
def showSpectrogram(Zxx, fs, figTitle, isLog=True):
    fig, ax = plt.subplots()
    print("Spectrogram shape: ", Zxx.shape)

    if isLog:
        img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(Zxx),ref=np.max),
                                    sr=fs, hop_length=512, fmax=fs/2,
                                    y_axis='linear', x_axis='time', ax=ax)
    else:
        img = librosa.display.specshow(np.abs(Zxx),
                                    sr=fs, hop_length=512, fmax=fs/2,
                                    y_axis='linear', x_axis='time', ax=ax)
    ax.set_title(figTitle)
    if isLog:
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    else:
        fig.colorbar(img, ax=ax, format="%+2.0f")
    # fig.set_figheight(5)
    # fig.set_figwidth(5)
    plt.show()

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


# [TODO]
# need a method to visualise the cues

if __name__ == "__main__":
    pass