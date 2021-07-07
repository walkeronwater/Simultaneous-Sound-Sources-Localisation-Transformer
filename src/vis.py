import matplotlib.pyplot as plt
import librosa
import librosa.display

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