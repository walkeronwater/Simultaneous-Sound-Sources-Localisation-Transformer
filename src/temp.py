import argparse

parser = argparse.ArgumentParser(description='Training hyperparamters')
parser.add_argument('dataDir', type=str, help='Directory of saved cues')
parser.add_argument('modelDir', type=str, help='Directory of model to be saved at')
parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
parser.add_argument('--numEnc', default=6, type=int, help='Number of encoder layers')
parser.add_argument('--numFC', default=3, type=int, help='Number of FC layers')
parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')
parser.add_argument('--numEpoch', default=30, type=int, help='Number of epochs')
parser.add_argument('--batchSize', default=32, type=int, help='Batch size')

args = parser.parse_args()
print(args.dataDir)
print(args.modelDir)
trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
print(trainValidSplit)
print(args.numEnc)
print(args.numFC)
print(args.valDropout)
print(args.numEpoch)
print(args.batchSize)
