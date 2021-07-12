# retrieve the model state with the best performance

valDropout = 0.3
num_layers = 6
model = SSSL(Nloc, Ntime, Nfreq, Ncues, num_layers, 8, device, 4, valDropout, False).to(device)
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
expName = "vol_144000_loc_24_layer_6/"
checkpointPath = rootDir+expName
model, optimizer, epoch = loadCheckpoint(model, optimizer, checkpointPath+"param.pth.tar")

lenSliceInSec = 0.5   # length of audio slice in sec
Nfreq = 512
Ntime = 44
Ncues = 5
Nloc = 24
Nsample = Nloc * 500

isDisk = False
# allocate tensors cues and labels in RAM
if not isDisk:
    if "cues_" in globals() or "cues_" in locals():
        del cues_
    if "labels_" in globals() or "labels_" in locals():
        del labels_
    cues_ = torch.zeros((Nsample, Nfreq, Ntime, Ncues))
    labels_ = torch.zeros((Nsample,))

# valSNRList = [-10,-5,0,5,10,15,20,100]
valSNRList = [5,10,15,20,100]

dirName = '/content/saved_cues/'

if not os.path.isdir(dirName):
    os.mkdir(dirName)

print(path)

for valSNR in valSNRList:
    fileCount = 0   # count the number of data samples
    for audioIndex in range(len(path)):
        if fileCount == Nsample:
            break
        # print("Audio index: ", audioIndex)
        audio, fs_audio = sf.read(path[audioIndex])
        # audio = librosa.resample(audio, fs_audio, fs_HRIR)

        audioSliceList = audioSliceGenerator(audio, fs_HRIR, lenSliceInSec)

        for sliceIndex in range(len(audioSliceList)):
            if fileCount == Nsample:
                break
            audioSlice = audio[audioSliceList[sliceIndex]]

            for locIndex in range(Nloc):
                if fileCount == Nsample:
                    break

                hrirLeft_re = librosa.resample(hrirSet[locIndex, 0], fs_HRIR, fs_audio)
                hrirRight_re = librosa.resample(hrirSet[locIndex, 1], fs_HRIR, fs_audio)
                sigLeft = np.convolve(audioSlice, hrirLeft_re)
                sigRight = np.convolve(audioSlice, hrirRight_re)

                # print("Location index: ", locIndex)
                # showSpectrogram(sigLeft, fs_HRIR)
                # showSpectrogram(sigRight, fs_HRIR)
            
                specLeft = calSpectrogram(sigLeft + noiseGenerator(sigLeft, valSNR))
                specRight = calSpectrogram(sigRight + noiseGenerator(sigRight, valSNR))

                ipdCues = calIPD(specLeft, specRight)
                ildCues = calILD(specLeft, specRight)
                r_l, theta_l  = cartesian2euler(specLeft)
                r_r, theta_r  = cartesian2euler(specRight)

                cues = concatCues([ipdCues, r_l, theta_l, r_r, theta_r], (Nfreq, Ntime))

                # save cues onto disk
                if isDisk:
                    saveCues(cues, locIndex, dirName, fileCount, locLabel, task="azim")
                else:
                    cues_[fileCount] = cues
                    labels_[fileCount] = locIndex2Label(locIndex, task="azim")

                '''if fileCount == 23:
                    raise SystemExit("Debugging")'''

                fileCount += 1
                # if fileCount % (Nloc*len(valSNRList)) == 0:
                #     print("# location set ("+str(Nloc*len(valSNRList))+" samples per set): ",
                #           fileCount // (Nloc*len(valSNRList)))

    # create tensor dataset from data loaded in RAM
    if not isDisk:
        dataset = TensorDataset(cues_, labels_.long())

        test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)

    '''
    testing
    '''
    # confusion matrix
    confusion_matrix = torch.zeros(Nloc, Nloc)

    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    # test phase
    model.isDebug=False
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            # print(inputs.shape)

            outputs = model(inputs)
            # print(outputs.shape)
            # print(labels.shape)
            test_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels.data).sum().item()
            
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('For SNR: %d Test_Loss: %.04f | Test_Acc: %.4f%% '
        % (valSNR, test_loss, 100.0 * test_correct / test_total))


# confusion matrix
confusion_matrix = torch.zeros(Nloc, Nloc)

test_loss = 0.0
test_correct = 0.0
test_total = 0.0
# test phase
model.isDebug=False
model.eval()
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        # print(inputs.shape)

        outputs = model(inputs)
        # print(outputs.shape)
        # print(labels.shape)
        test_loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels.data).sum().item()
        
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print('Test_Loss: %.04f | Test_Acc: %.4f%% '
    % (test_loss, 100.0 * test_correct / test_total))