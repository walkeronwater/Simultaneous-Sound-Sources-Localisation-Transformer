import numpy as np
from scipy import signal

# method to generate audio slices for a given length requirement
# with a hardcoded power threshold
def audioSliceGenerator(audioSeq, sampleRate, lenSliceInSec, isDebug=False):
    lenAudio = audioSeq.size
    lenSlice = round(sampleRate * lenSliceInSec)
    # audioSliceList = [range(lenSlice*i, lenSlice *(i+1)) for i in range(lenAudio//lenSlice)]
    # print(len(audioSliceList))
    # print(lenAudio//lenSlice)

    audioSliceList = []
    if isDebug:
        powerList = []
    # threshold for spectrum power
    for i in range(lenAudio//lenSlice):
        sliced = audioSeq[lenSlice*i:lenSlice *(i+1)]
        # print("slice power", np.mean(np.power(sliced, 2)))
        if isDebug:
            powerList.append(np.mean(np.power(sliced, 2)))
        if np.mean(np.power(sliced, 2)) > 0.01:
            audioSliceList.append(range(lenSlice*i, lenSlice *(i+1)))
    if isDebug:
        return audioSliceList, powerList
    else:
        return audioSliceList

# visualise
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

def noiseGenerator(sigSeq, valSNR):
    # assert debug
    # assert (
    #     valSNR >= 20
    # ), "Input data needs to be reshaped to (1, length of sequence)"

    if -10 <= valSNR <= 20:
        sigSeqPower = 10*np.log10(np.mean(np.power(sigSeq, 2)))
        noiseSeqPower = np.power(10, (sigSeqPower - valSNR)/10)
        noiseSeq = np.random.normal(0, np.sqrt(noiseSeqPower), sigSeq.shape)
        del sigSeqPower, noiseSeqPower
        return noiseSeq
    else:
        return 0

def cartesian2euler(val):
    x = val.real
    y = val.imag
    
    r = np.sqrt(x**2+y**2)

    theta = np.arctan(
        np.divide(y, x, where=x!=0)
    )
    # if x != 0:
    #     theta = np.arctan(y/x)
    # else:
    #     theta = np.pi/2
        
    return r, theta

def calIPD(seqL, seqR):
    ipd = np.angle(np.divide(seqL, seqR, out=np.zeros_like(seqL), where=np.absolute(seqR)!=0))
    return ipd

def calILD(seqL, seqR):
    ild = 20*np.log10(np.divide(np.absolute(seqL), np.absolute(seqR), out=np.zeros_like(seqL), where=np.absolute(seqR)!=0))
    return ild
    
# [TODO]
# method to normalise a sequence which can be broadcasted to a sequence of sequence
def normalise(seq):
    # return (seq - np.mean(seq))/(np.std(seq))
    return seq/np.linalg.norm(seq)

def binauralCues(sigPair, fs, valSNR):
    if len(sigPair.shape) == 4:
        print("Warning: high memory requirement in binauralCues()")
        f, t, Zxx = signal.stft(sigPair[0, 0, 0], fs, nperseg=1023)
        # spectralCues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]) + (4,), dtype='float')
        # ipdCues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]), dtype='float')
        cues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]) + (5,), dtype='float')

        del f, t, Zxx

        for i in range(sigPair.shape[0]):
            for locIndex in range(sigPair.shape[1]):
                
                _, _, Zxx_l = signal.stft(
                    sigPair[i, locIndex, 0] 
                    + noiseGenerator(sigPair[i, locIndex, 0], valSNR) # noise added
                    , fs, nperseg = 1023
                )              
                                            
                _, _, Zxx_r = signal.stft(
                    sigPair[i, locIndex, 1] 
                    + noiseGenerator(sigPair[i, locIndex, 1], valSNR) # noise added
                    , fs, nperseg = 1023
                )
                # print(Zxx_l.shape)
                # print(Zxx_r.shape)

                r_l, theta_l = cartesian2euler(Zxx_l)
                r_r, theta_r = cartesian2euler(Zxx_r)

                # ipdCues[i, locIndex] = normalise(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0)))
                # spectralCues[i, locIndex] = np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
                cues[i, locIndex] = np.concatenate(
                    (
                        np.expand_dims(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0)), axis=-1),
                        np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
                    ),
                    axis=-1
                )
        
        del Zxx_l, Zxx_r
        return cues
    elif len(sigPair.shape) == 2:
        _, _, Zxx_l = signal.stft(
            sigPair[0] 
            + noiseGenerator(sigPair[0], valSNR) # noise added
            , fs, nperseg = 1023
        )              
                                    
        _, _, Zxx_r = signal.stft(
            sigPair[1] 
            + noiseGenerator(sigPair[1], valSNR) # noise added
            , fs, nperseg = 1023
        )
        # print(Zxx_l.shape)
        # print(Zxx_r.shape)

        r_l, theta_l = cartesian2euler(Zxx_l)
        r_r, theta_r = cartesian2euler(Zxx_r)
        
        # ipdCues[i, locIndex] = normalise(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0)))
        # spectralCues[i, locIndex] = np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
        cues = np.concatenate(
            (
                np.expand_dims(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0)), axis=-1),
                np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
            ),
            axis=-1
        )
    
        del Zxx_l, Zxx_r
        return cues

