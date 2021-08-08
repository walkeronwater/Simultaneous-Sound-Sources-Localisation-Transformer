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

class SourcePrediction:
    """
    Store the prediction for a single sound source
    """
    def __init__(self):
        self.elev_pred = []
        self.elev_target = []
        self.azim_pred = []
        self.azim_target = []
            
    def __call__(self, outputs, labels):
        """
        Args:
            outputs: shape of (batch size, 2)
            labels: shape of (batch size, 2)
        """
        
        self.elev_pred.extend([radian2Degree(i) for i in outputs[:, 0].tolist()])
        self.elev_target.extend([radian2Degree(i) for i in labels[:, 0].tolist()])
        self.azim_pred.extend([radian2Degree(i) for i in outputs[:, 1].tolist()])
        self.azim_target.extend([radian2Degree(i) for i in labels[:, 1].tolist()])

        # for i in range(outputs.shape[0]):
        #     self.elev_pred.append(outputs[i, 0].item())
        #     self.elev_target.append(labels[i, 0].item())
        #     self.azim_pred.append(outputs[i, 1].item())
        #     self.azim_target.append(labels[i, 1].item())


class VisualisePrediction:
    """
    Support multiple sound-source prediction
    """
    def __init__(self, Nsound):
        self.sound_list = []
        for i in range(Nsound):
            self.sound_list.append(SourcePrediction())
        self.Nsound = Nsound

    def __call__(self, outputs, labels):
        """
        Args:
            outputs: shape of (batch size, 2*Nsound)
            labels: shape of (batch size, 2*Nsound)
        """
        for i in range(self.Nsound):
            self.sound_list[i](outputs[:,2*i:2*(i+1)], labels[:,2*i:2*(i+1)])

    def report(self):
        # print(f"{len(self.elev_pred)}, {len(self.elev_target)}, {len(self.azim_pred)}, {len(self.azim_pred)}")
        for i in range(self.Nsound):
            x = np.linspace(-45, 90, 100)
            y = x
            plt.figure()
            plt.scatter(self.sound_list[i].elev_target, self.sound_list[i].elev_pred, color='blue')
            plt.plot(x, y,'-r')
            plt.xticks(range(-45, 91, 15))
            plt.yticks(range(-45, 91, 15))
            plt.xlabel("Ground truth")
            plt.ylabel("Prediction")
            plt.title(f"Elevation of sound source {i}")
            plt.show()

            x = np.linspace(0, 345, 100)
            y = x
            plt.figure()
            plt.scatter(self.sound_list[i].azim_target, self.sound_list[i].azim_pred, color='blue')
            plt.plot(x, y,'-r')
            plt.xticks(range(0, 360, 30))
            plt.yticks(range(0, 360, 30))
            plt.xlabel("Ground truth")
            plt.ylabel("Prediction")
            plt.title(f"Azimuth of sound source {i}")
            plt.show()

            plt.figure()
            plt.scatter(self.sound_list[i].azim_target, self.sound_list[i].elev_target, color='red')
            plt.scatter(self.sound_list[i].azim_pred, self.sound_list[i].elev_pred, color='blue')
            plt.xticks(range(0, 360, 30))
            plt.yticks(range(-45, 91, 15))
            plt.xlabel("Elevation")
            plt.ylabel("Azmiuth")
            plt.title(f"Prediction and target of sound source {i}")
            plt.show()

    def plotRegression(self, pred_list, target_list):
        pass

def createTestSet(loc_idx_1, loc_idx_2):
    audio_index_1 = 0
    audio_index_2 = 0
    src_1 = AudioSignal(path=src_1_path[audio_index_1], slice_duration=1)
    src_2 = AudioSignal(path=src_2_path[audio_index_2], slice_duration=1)
    # binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=src_1.fs_audio)
    # binaural_cues = BinauralCues(fs_audio=src_1.fs_audio, prep_method="standardise")

    slice_idx_1 = 0
    slice_idx_2 = 0
    count = 0
    while True:
        # print(f"Current audio (src 1): {audio_index_1}, and (src 2): {audio_index_2}")
        # print(f"Number of slices (audio 1): {len(src_1.slice_list)}, and (audio 2): {len(src_2.slice_list)}")
        if slice_idx_1 >= len(src_1.slice_list):
            slice_idx_1 = 0
            audio_index_1 += 1
            src_1 = AudioSignal(path=src_1_path[audio_index_1], slice_duration=1)
            
        if slice_idx_2 >= len(src_2.slice_list):
            slice_idx_2 = 0
            audio_index_2 += 1
            src_2 = AudioSignal(path=src_2_path[audio_index_2], slice_duration=1)
        

        sig_sliced_1 = src_1(idx=slice_idx_1)
        sig_sliced_2 = src_2(idx=slice_idx_2)

        sigL_1, sigR_1 = binaural_sig(sig_sliced_1, loc_idx_1)
        sigL_2, sigR_2 = binaural_sig(sig_sliced_2, loc_idx_2)
        magL, phaseL, magR, phaseR = binaural_cues(sigL_1+sigL_2, sigR_1+sigR_2)

        # save_cues(cuesList=[magL, phaseL, magR, phaseR], locIndex=[loc_idx_1, loc_idx_2])
        # if save_cues.fileCount == args.Nsample:
        #     return
        test_cues[count] = torch.tensor([magL, phaseL, magR, phaseR]).permute(1,2,0)

        test_label[count] = torch.from_numpy(degree2Radian(np.concatenate((locLabel[loc_idx_1], locLabel[loc_idx_2]), axis=-1)))

        count += 1
        if count >= Nsample:
            return

        slice_idx_1 += 1
        slice_idx_2 += 1
        # print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing phase')
    parser.add_argument('dataDir', type=str, help='Directory of saved cues')
    parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
    parser.add_argument('whichModel', type=str, help='whichModel?')
    parser.add_argument('--isHPC', default="False", type=str, help='isHPC?')
    parser.add_argument('--isDebug', default="False", type=str, help='isDebug?')
    args = parser.parse_args()
    """define Nfreq, Ntime, Ncues"""
    Nfreq = 512
    Ntime = 72
    Ncues = 4
    Nsound = 2
    task = "allRegression"
    whichDec = args.whichModel
    isHPC = True if args.isHPC.lower()[0] == "t" else False
    isDebug = True if args.isDebug.lower()[0] == "t" else False
    num_workers = 0
    Nsample = 2
    batch_size = 32
    # model_dir = "D:/SSSL-D/HPC/0808_2Sound_ea/"
    data_dir = args.dataDir
    model_dir = args.modelDir
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
        # numEnc=args.numEnc,
        # numFC=args.numFC,
    )
    # model, val_optim = loadCheckpoint(
    #     model=model, optimizer=None, scheduler=None,
    #     loadPath=model_dir,
    #     task=task, phase="test", whichBest="bestValLoss"
    # )

    if isHPC:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    checkpoint = torch.load(loadPath+"param_bestValLoss.pth.tar")
    model.load_state_dict(checkpoint['model'], strict=False)

    cost_func = CostFunc(task=task, Nsound=Nsound, device=device)

    """mix sound sources"""
    # loc_idx_1 = 0
    # loc_idx_2 = 0
    src_1_path = glob(os.path.join("./audio_train/speech_male/*"))
    src_2_path = glob(os.path.join("./audio_train/speech_female/*"))
    src_1 = AudioSignal(path=src_1_path[0], slice_duration=1)
    binaural_sig = BinauralSignal(hrir=hrirSet, fs_hrir=fs_HRIR, fs_audio=src_1.fs_audio)
    binaural_cues = BinauralCues(fs_audio=src_1.fs_audio, prep_method="standardise")
    loc_region = LocRegion(locLabel=locLabel)
    vis_pred = VisualisePrediction(Nsound=Nsound)
    for loc_idx_1 in loc_region.high_left + loc_region.low_left:
        for loc_idx_2 in loc_region.high_right + loc_region.low_right:
            print(f"Test set created for location pair: {loc_idx_1}, {loc_idx_2}")
            createTestSet(loc_idx_1, loc_idx_2)

            dataset = TensorDataset(test_cues, test_label)

            # test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            valid_dataset = CuesDataset(data_dir + "/valid/",
                                task, Nsound, locLabel, isDebug=False)

            isPersistent = True if num_workers > 0 else False
            test_loader = MultiEpochsDataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                persistent_workers=isPersistent
            )
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
                    print(f"RMS angle error of two sources (over one batch): {torch.mean(radian2Degree(cost_func(outputs, labels))).item():.2f}")
                    vis_pred(outputs, labels)
                    test_loss = cost_func(outputs, labels)
                    test_sum_loss += test_loss.item()
                test_loss = test_sum_loss / (i + 1)
                print('Test Loss: %.04f | Test Acc: %.04f '
                    % (test_loss, test_acc))
            
        vis_pred.report()