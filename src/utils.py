def audioSliceGenerator(audioSeq, sampleRate, lenSliceInSec):
    lenAudio = audioSeq.size
    lenSlice = round(sampleRate * lenSliceInSec)
    # audioSliceList = [range(lenSlice*i, lenSlice *(i+1)) for i in range(lenAudio//lenSlice)]
    # print(len(audioSliceList))
    # print(lenAudio//lenSlice)

    audioSliceList = []
    # threshold for spectrum power
    for i in range(lenAudio//lenSlice):
        sliced = audioSeq[lenSlice*i:lenSlice *(i+1)]
        # print("slice power", np.mean(np.power(sliced, 2)))
        if np.mean(np.power(sliced, 2)) > 0.01:
            audioSliceList.append(range(lenSlice*i, lenSlice *(i+1)))

    return audioSliceList


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
        
    return normalise(r), normalise(theta)

def calIPD(seqL, seqR):
    temp = np.divide(seqL, seqR, out=np.zeros_like(seqL), where=seqR!=0)
    ipd = np.arctan(np.divide(np.imag(temp), np.real(temp), out=np.zeros_like(np.real(temp)), where=np.real(temp)!=0))
    del temp
    return normalise(ipd)


# method to normalise a sequence which can be broadcasted to a sequence of sequence [TODO]
def normalise(seq):
    return (seq - np.mean(seq))/(np.std(seq))

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

