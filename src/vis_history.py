import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import csv
import pandas as pd
from glob import glob
import torch

def loadHistory(loadPath, figPath, isDebug):
    trainHistory = glob(os.path.join(loadPath, "curve*"))

    history = {
        'train_acc': [],
        'train_loss': [],
        'valid_acc': [],
        'valid_loss': []
    }
    for i in range(len(trainHistory)):
        checkpt = torch.load(
            loadPath+"curve_epoch_"+str(i+1)+".pth.tar"
        )
        for idx in history.keys():
            history[idx].append(round(checkpt[idx], 5))
    if isDebug:
        for idx in history.keys(): 
            print(idx, history[idx])

    # plt.subplot(211, figs)
    # fig = matplotlib.pyplot.gcf()
    # figure(figsize=(8, 6), dpi=80)
    plt.plot(range(1, len(history['train_loss'])+1), history['train_loss'])
    plt.plot(range(1, len(history['train_loss'])+1), history['valid_loss'])
    plt.xlabel('epoch')
    plt.xticks(range(1, len(history['train_loss'])+1))
    plt.ylabel('Loss')
    plt.title('Loss curve')
    plt.legend(['Train', 'Valid'])
    plt.grid()
    plt.savefig(figPath+"loss.png")
    # plt.show()
    plt.close()

    # plt.subplot(212)
    # plt.plot(range(1, len(history['train_acc'])+1), history['train_acc'])
    # plt.plot(range(1, len(history['train_acc'])+1), history['valid_acc'])
    # plt.xlabel('epoch')
    # plt.xticks(range(1, len(history['train_acc'])+1))
    # plt.ylabel('Accuracy (%)')
    # plt.title('Accuracy curve')
    # plt.legend(['Train', 'Valid'])
    # plt.grid()
    # plt.savefig(figPath+"acc.png")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training history plot')
    parser.add_argument('modelDir', type=str, help='Directory of model')
    parser.add_argument('--prefixName', default='None', type=str, help='Prefix name')
    parser.add_argument('--figDir', type=str, help='Directory of figures to be saved at')
    parser.add_argument('--isDebug', type=str, default="False", help='isDebug?')
    
    args = parser.parse_args()
    # if args.modelDir[-1] != "/":
    #     args.modelDir += "/"
    if args.figDir == None:
        args.figDir = args.modelDir
    else:
        if args.figDir[-1] != "/":
            args.figDir += "/"
    if args.isDebug == "True":
        args.isDebug = True
    else:
        args.isDebug = False
    
    print("Model directory: ", args.modelDir)
    
    if args.prefixName.lower() != "None":


        path = glob(args.modelDir+args.prefixName+"*")

        path = [
            d for d in os.listdir(args.modelDir) if os.path.isdir(os.path.join(args.modelDir, d))
            and d[:len(args.prefixName)].lower()==args.prefixName
        ]

        for pth in path:
            print(args.modelDir+pth)
            try:
                loadHistory(args.modelDir+pth+"/", args.modelDir+pth+"/", args.isDebug)
            except:
                pass
    else:
        loadHistory(args.modelDir+"/", pth+"/", args.isDebug)