import argparse
from glob import glob
import os
import soundfile as sf
import torch
import torch.nn as nn

from load_data import *
from utils import *
from utils_model import *
from models import *
from loss import *


def createTestSet(loc_idx, val_SNR):
    audio_index = 0
    src = AudioSignal(path=src_path[audio_index], slice_duration=1)
    # binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=src.fs_audio)
    # binaural_cues = BinauralCues(fs_audio=src.fs_audio, prep_method="standardise")

    slice_idx = 0
    count = 0
    while True:
        # print(f"Current audio (src 1): {audio_index}, and (src 2): {audio_index_2}")
        # print(f"Number of slices (audio 1): {len(src.slice_list)}, and (audio 2): {len(src_2.slice_list)}")
        if slice_idx >= len(src.slice_list):
            slice_idx = 0
            audio_index += 1
            src = AudioSignal(path=src_path[audio_index], slice_duration=1)

        sig_sliced = src(idx=slice_idx)
        binaural_sig.val_SNR = val_SNR
        
        sigL, sigR = binaural_sig(sig_sliced, loc_idx)
        magL, phaseL, magR, phaseR = binaural_cues(sigL, sigR)

        # save_cues(cuesList=[magL, phaseL, magR, phaseR], locIndex=[loc_idx])
        # if save_cues.fileCount == args.Nsample:
        #     return
        test_cues[count] = torch.tensor([magL, phaseL, magR, phaseR]).permute(1,2,0)

        test_label[count] = torch.from_numpy(degree2Radian(locLabel[loc_idx]))

        count += 1
        if count >= Nsample:
            return

        slice_idx += 1
        # print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing phase')
    parser.add_argument('dataDir', type=str, help='Directory of audio files')
    parser.add_argument('audioDir', type=str, help='Directory of audio files')
    parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
    parser.add_argument('whichModel', type=str, help='whichModel?')
    parser.add_argument('whichDec', type=str, help='Which decoder')
    
    parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    args = parser.parse_args()

    """define Nfreq, Ntime, Ncues"""
    Nfreq = 512
    Ntime = 72
    Ncues = 4
    Nsound = 1
    task = "allRegression"
    whichDec = args.whichDec
    audio_dir = args.audioDir
    model_dir = args.modelDir
    isHPC = True if args.isHPC.lower()[0] == "t" else False
    isDebug = True if args.isDebug.lower()[0] == "t" else False
    num_workers = 0
    Nsample = 1
    batch_size = 32
    # model_dir = "D:/SSSL-D/HPC/0608_1Sound_ea/"
    model_dir = args.modelDir
    valSNRList = [-5,0,5,10,15,20,25,30,35]

    # path = args.hrirDir + "/IRC*"
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)
    
    # save_cues = SaveCues(savePath=args.cuesDir+"/", locLabel=locLabel)
    """create a tensor that stores all testing examples"""
    test_cues = torch.empty((Nsample, Nfreq, Ntime, Ncues))
    test_label = torch.empty((Nsample, 2*Nsound))

    """load model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        task=task,
        Ntime=Ntime,
        Nfreq=Nfreq,
        Ncues=Ncues,
        Nsound=Nsound,
        whichEnc="diy",
        whichDec=whichDec,
        device=device,
        numEnc=args.numEnc
        # numFC=args.numFC,
    )
    if isHPC:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    
    # model, val_optim = loadCheckpoint(
    #     model=model, optimizer=None, scheduler=None,
    #     loadPath=model_dir,
    #     task=task, phase="test", whichBest="bestValLoss"
    # )

    checkpoint = torch.load(model_dir+"param_bestValLoss.pth.tar")
    model.load_state_dict(checkpoint['model'], strict=True)

    cost_func = CostFunc(task=task, Nsound=Nsound, device=device, coordinates=args.coordinates)

    """mix sound sources with noise"""
    src_path = glob(os.path.join(audio_dir+"/*"))
    src = AudioSignal(path=src_path[0], slice_duration=1)
    binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=src.fs_audio)
    binaural_cues = BinauralCues(fs_audio=src.fs_audio, prep_method="standardise")
    loc_region = LocRegion(locLabel=locLabel)
    
    for val_SNR in valSNRList:
        confusion = Confusion()
        vis_pred = VisualisePrediction(Nsound=Nsound)
        for loc_idx in range(0,168,1):
            print(f"Test set created for location: {loc_idx}, with SNR {val_SNR}")
            createTestSet(loc_idx, val_SNR)

            dataset = TensorDataset(test_cues, test_label)

            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            test_correct = 0.0
            test_total = 0.0
            test_sum_loss = 0.0
            test_loss = 0.0
            test_acc = 0.0
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(test_loader, 0):
                    inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                    outputs = model(inputs)

                    if isDebug:
                        print(
                            "Input shape: ", inputs.shape, "\n",
                            "label shape: ", labels.shape, "\n",
                            "labels: ", labels[:5], "\n",
                            "Output shape: ", outputs.shape, "\n",
                            "Outputs: ", outputs[:5]
                        )
                    print(f"RMS angle error (over one batch): {torch.mean(radian2Degree(cost_func.calDoALoss(outputs, labels))).item():.2f}")
                    
                    test_loss = cost_func(outputs, labels)
                    vis_pred(
                        outputs, labels,
                        [loc_idx],
                        [torch.mean(radian2Degree(cost_func.calDoALoss(outputs[:, 0:2], labels[:, 0:2]))).item()]
                    )
                    test_sum_loss += test_loss.item()
                test_loss = test_sum_loss / (i + 1)
                print('Test Loss: %.04f | Test Acc: %.04f '
                    % (test_loss, test_acc))
            
        # vis_pred.report()   
        vis_pred.report(
            fixed_src = locLabel[loc_idx],
            # path = args.plotDir
        )