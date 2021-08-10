import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast visualise csv files')
    parser.add_argument('csvPath', type=str, help='Path to the csv file')

    args = parser.parse_args()
    print("Path to the csv file: ", args.csvPath)

    if args.csvPath[-4:] != ".csv":
        args.csvPath += ".csv"
        
    csvF = pd.read_csv(args.csvPath, header=None)
    print(csvF)