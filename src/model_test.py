import argparse
from glob import glob
import os
import soundfile as sf
import torch
import torch.nn as nn

from data_loader import *
from utils import *
from utils_train import *
from utils_test import *
from models import *
from model_CRNN import *
from loss import *


def createTestSet(loc_idx, val_SNR):
    audio_index = 0
    src = AudioSignal(path=src_path[audio_index], slice_duration=1)
    # binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=src.fs_audio)
    # binaural_cues = BinauralCues(fs_audio=src.fs_audio, prep_method="standardise")

    slice_idx = 0
    count = 0
    while True:
        # print(f"Current audio: {audio_index}")
        # print(f"Number of slices: {len(src.slice_list)}")
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

        test_label[count] = torch.from_numpy(degree2Radian(load_hrir.loc_label[loc_idx]))

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
    
    parser.add_argument('--numEnc', default=3, type=int, help='Number of encoder layers')
    parser.add_argument('--numFC', default=4, type=int, help='Number of FC layers')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    parser.add_argument('--coordinates', default="spherical", type=str, help='Spherical or Cartesian')
    parser.add_argument('--isLogging', default="False", type=str, help='Log down prediction in a csv file')
    parser.add_argument('--logName', default="test_log_1Sound", type=str, help='Log down prediction in a csv file')
    parser.add_argument('--hrtfDir', default="None", type=str, help='HRTF dir')

    args = parser.parse_args()
    """check input directories end up with /"""
    dir_var = {
        "data": args.dataDir,
        "audio": args.audioDir,
        "model": args.modelDir
    }
    for idx in dir_var.keys():
        dir_var[idx] += "/"
    if not os.path.isdir(dir_var["model"]):
        raise SystemExit("Model not found.")

    """create dicts holding the directory and flag variables"""
    flag_var = {
        "isDebug": args.isDebug,
        "isHPC": args.isHPC,
        "isLogging": args.isLogging
    }
    for idx in flag_var.keys():
        flag_var[idx] = True if flag_var[idx][0].lower() == "t" else False

    """define Nfreq, Ntime, Ncues"""
    Nfreq = 512
    Ntime = 30
    Ncues = 4
    Nsound = 1
    task = "allRegression"
    whichDec = args.whichDec
    num_workers = 0
    Nsample = 8
    batch_size = 32
    valSNRList = [-5,0,5,10,15,20,25,30,35]

    # path = args.hrirDir + "/IRC*"
    path = "./HRTF/IRC*"
    # hrirSet, locLabel, fs_HRIR = loadHRIR(path=args.hrtfDir+"/IRC*")
    load_hrir = LoadHRIR(path=args.hrtfDir+"/IRC*")

    loss_all_subject = []
    loss_all_subject_x = []
    for subject_idx in range(load_hrir.num_subject):
        load_hrir.load_subject(subject_idx)
        
        # save_cues = SaveCues(savePath=args.cuesDir+"/", locLabel=locLabel)
        """create a tensor that stores all testing examples"""
        test_cues = torch.empty((Nsample, Nfreq, Ntime, Ncues))
        test_label = torch.empty((Nsample, 2*Nsound))

        """load model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.whichModel.lower() == "transformer":
            model = TransformerModel(
                task=task,
                Ntime=Ntime,
                Nfreq=Nfreq,
                Ncues=Ncues,
                Nsound=Nsound,
                whichEnc="diy",
                whichDec=args.whichDec,
                device=device,
                numEnc=args.numEnc,
                coordinates=args.coordinates,
                forward_expansion=4,
                numFC=args.numFC,
            )
            print("model: transformer")
        elif args.whichModel.lower() == "crnn":
            model = CRNN(
                task=task,
                Ntime=Ntime,
                Nfreq=Nfreq,
                Ncues=Ncues,
                Nsound=Nsound,
                whichDec="src",
                num_conv_layers=4,
                num_recur_layers=2,
                num_FC_layers=args.numFC,
                device=device,
                isDebug=False,
                coordinates="spherical"
            )
            print("model: CRNN")
        else:
            raise SystemExit("Unsupported model.")
        if flag_var["isHPC"]:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        
        # model, val_optim = loadCheckpoint(
        #     model=model, optimizer=None, scheduler=None,
        #     loadPath=model_dir,
        #     task=task, phase="test", whichBest="bestValLoss"
        # )

        checkpoint = torch.load(dir_var["model"] + "param_bestValLoss.pth.tar")
        model.load_state_dict(checkpoint['model'], strict=True)

        cost_func = CostFunc(task=task, Nsound=Nsound, device=device, coordinates=args.coordinates)

        csv_flag = False
        csv_name = dir_var['model'] + args.logName + ".csv"
        """mix sound sources with noise"""
        src_path = glob(os.path.join(dir_var["audio"] + "/*"))
        src = AudioSignal(path=src_path[0], slice_duration=1)
        binaural_sig = BinauralSignal(hrir=load_hrir.hrir_set, fs_hrir=fs_HRIR, fs_audio=src.fs_audio)
        binaural_cues = BinauralCues(fs_audio=src.fs_audio, prep_method="minmax")
        loc_region = LocRegion(loc_label=load_hrir.loc_label)
        
        loss_allSNR = 0.0
        loss_allSNR_count = 0
        fb, lr, ud, error = [], [], [], []
        for val_SNR in valSNRList:
            confusion = Confusion()
            vis_pred = VisualisePrediction(Nsound=Nsound)
            count = 0
            for loc_idx in range(0,186,1):
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
                    for batch_i, (inputs, labels) in enumerate(test_loader, 0):
                        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
                        outputs = model(inputs)

                        if flag_var["isDebug"]:
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

                        if flag_var["isLogging"]:
                            if not csv_flag:
                                csv_flag = True
                                with open(csv_name, 'w') as csvFile:
                                    for batch_idx in range(outputs.shape[0]):
                                        csvFile.write(str(val_SNR))
                                        csvFile.write(',')
                                        csvFile.write(str(loc_idx))
                                        csvFile.write(',')
                                        for i in range(outputs.shape[1]):
                                            csvFile.write(str(radian2Degree(outputs[batch_idx, i].item())))
                                            csvFile.write(',')
                                        for i in range(outputs.shape[1]):
                                            csvFile.write(str(radian2Degree(labels[batch_idx, i].item())))
                                            csvFile.write(',')
                                        csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 0:2].unsqueeze(0), labels[batch_idx, 0:2].unsqueeze(0)).item())))
                                        csvFile.write('\n')
                            else:
                                with open(csv_name, 'a') as csvFile:
                                    for batch_idx in range(outputs.shape[0]):
                                        csvFile.write(str(val_SNR))
                                        csvFile.write(',')
                                        csvFile.write(str(loc_idx))
                                        csvFile.write(',')
                                        for i in range(outputs.shape[1]):
                                            csvFile.write(str(radian2Degree(outputs[batch_idx, i].item())))
                                            csvFile.write(',')
                                        for i in range(outputs.shape[1]):
                                            csvFile.write(str(radian2Degree(labels[batch_idx, i].item())))
                                            csvFile.write(',')
                                        csvFile.write(str(radian2Degree(cost_func.calDoALoss(outputs[batch_idx, 0:2].unsqueeze(0), labels[batch_idx, 0:2].unsqueeze(0)).item())))
                                        csvFile.write('\n')
                        confusion(outputs, labels)
                        count += outputs.shape[0]
                    # print(f"UD: {confusion.se_UD}, LR: {confusion.se_LR}, FB: {confusion.se_FB}")
                    test_loss = test_sum_loss / (batch_i + 1)
                    loss_allSNR += test_loss
                    loss_allSNR_count += 1
                    test_acc = radian2Degree(test_loss)
                    # print('Test Loss: %.04f | RMS angle error in degree: %.04f '
                    #     % (test_loss, test_acc))

            # error.append(radian2Degree())
            fb.append(radian2Degree(np.sqrt(confusion.se_FB/count)))
            ud.append(radian2Degree(np.sqrt(confusion.se_UD/count)))
            lr.append(radian2Degree(np.sqrt(confusion.se_LR/count)))
            print("UD: ", radian2Degree(np.sqrt(confusion.se_UD/count)))
            print("FB: ", radian2Degree(np.sqrt(confusion.se_FB/count)))
            print("LR: ", radian2Degree(np.sqrt(confusion.se_LR/count)))
            
            # vis_pred.report(
            #     fixed_src = locLabel[loc_idx],
            #     # path = args.plotDir
            # )
        """Plot UD FB LR confusion"""
        plt.plot(valSNRList, fb)
        plt.plot(valSNRList, lr)
        plt.plot(valSNRList, ud)
        plt.xlabel("SNR")
        plt.ylabel("RMS angle error in degree")
        plt.legend(["Front-Back", "Left-right", "High-low"])
        plt.title("Confusion plots")
        plt.grid()
        plt.show()
                        
        loss_all_subject.append(loss_allSNR/loss_allSNR_count)
        loss_all_subject_x.append(subject_idx)