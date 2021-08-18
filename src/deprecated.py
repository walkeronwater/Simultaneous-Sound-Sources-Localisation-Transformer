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



# method to generate audio slices for a given length requirement
# with a hardcoded power threshold
def audioSliceGenerator(audioSeq, sampleRate, lenSliceInSec, threshold=0.01, isDebug=False):
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
        if np.mean(np.power(sliced, 2)) > threshold:
            audioSliceList.append(range(lenSlice*i, lenSlice *(i+1)))
    if isDebug:
        return audioSliceList, powerList
    else:
        return audioSliceList

# method to generate a sequence of noise for a given SNR
def noiseGenerator(sigSeq, valSNR):
    # assert debug
    # assert (
    #     valSNR >= 20
    # ), "Input data needs to be reshaped to (1, length of sequence)"
    if valSNR >= 100:
        return 0
    else:
        sigSeqPower = 10*np.log10(np.mean(np.power(sigSeq, 2)))
        noiseSeqPower = np.power(10, (sigSeqPower - valSNR)/10)
        noiseSeq = np.random.normal(0, np.sqrt(noiseSeqPower), sigSeq.shape)
        del sigSeqPower, noiseSeqPower
        return noiseSeq

'''def addNoise(sigPair):
    valSNR = 1
    # loop through all training examples
    for i in range(sigPairList[0].shape[0]):
        # loop through all locations
        for locIndex in range(sigPairList[0].shape[1]):
            noiseLeft = noiseGenerator(np.expand_dims(sigPairList[0][i,locIndex,0], axis=0), valSNR)
            noiseRight = noiseGenerator(np.expand_dims(sigPairList[0][i,locIndex,1], axis=0), valSNR)'''

def calSpectrogram(seq, fs):
    Nfft = 1023
    Zxx = librosa.stft(seq, n_fft=Nfft, hop_length=512)
    return Zxx

# utility methods for binaural cue extraction
def cartesian2euler(spec):
    # x = seq.real
    # y = seq.imag
    # r = np.sqrt(x**2+y**2)
    # theta = np.angle(np.divide(y, x, where=x!=0))
    # theta = np.arctan(
    #     np.divide(y, x, where=x!=0)
    # )
    # return r, np.unwrap(theta)
    mag = np.abs(spec)
    phase = np.angle(spec)
    phase = unwrap_phase(phase)
    return mag, phase

def calIPD(specL, specR):
    # ipd = np.angle(np.divide(seqL, seqR, out=np.zeros_like(seqL), where=np.absolute(seqR)!=0))
    ipd = np.angle(np.divide(specL, specR, out=np.zeros_like(specL), where=np.absolute(specR)!=0))
    ipd = unwrap_phase(ipd)
    return ipd

def calIPD_unwrap(seqL, seqR):
    ipd = np.angle(np.divide(seqL, seqR, out=np.zeros_like(seqL), where=np.absolute(seqR)!=0))
    return unwrap_phase(ipd)

def calILD(seqL, seqR):
    ild = 20*np.log10(np.divide(np.absolute(seqL), np.absolute(seqR), out=np.zeros_like(np.absolute(seqL)), where=np.absolute(seqR)!=0))
    return ild

# save cues as pt files and write labels in a csv file
def saveCues(cues, locIndex, dirName, fileCount, locLabel):
    if fileCount == 0:
        if os.path.isfile(dirName+'dataLabels.csv'):
            print("Directory exists -- overwriting")
            # if input('Delete saved_cues? ') == 'y':
            #     print('ok')
            shutil.rmtree(dirName)
            os.mkdir(dirName)
        
        with open(dirName+'dataLabels.csv', 'w') as csvFile:
            csvFile.write(str(fileCount))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "allClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimRegression")))
            csvFile.write('\n')
    else:
        with open(dirName+'dataLabels.csv', 'a') as csvFile:
            csvFile.write(str(fileCount))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "allClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimClass")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "elevRegression")))
            csvFile.write(',')
            csvFile.write(str(locIndex2Label(locLabel, locIndex, "azimRegression")))
            csvFile.write('\n')
    torch.save(cues, dirName+str(fileCount)+'.pt')

def concatCues(cuesList: list, cuesShape: tuple):
    lastDim = len(cuesList)
    cues = torch.zeros(cuesShape+(lastDim,), dtype=torch.float)

    for i in range(lastDim):
        cues[:,:,i] = torch.from_numpy(cuesList[i])

    return cues


