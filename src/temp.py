import argparse

parser = argparse.ArgumentParser(description='Training hyperparamters')
parser.add_argument('rootDir', type=str, help='Directory that the model will be saved at')
parser.add_argument('expName', type=str, help='The experiment name')
parser.add_argument('--trainValidSplit', default="0.8, 0.2", type=str, help='Training Validation split')
parser.add_argument('--numEnc', default=6, type=int, help='The number of encoder layers')
parser.add_argument('--numFC', default=3, type=int, help='The number of FC layers')
parser.add_argument('--valDropout', default=0.3, type=float, help='Dropout value')

args = parser.parse_args()
print(args.rootDir)
print(args.expName)
trainValidSplit = [float(item) for item in args.trainValidSplit.split(',')]
print(trainValidSplit)
print(args.numEnc)
print(args.numFC)
print(args.valDropout)
