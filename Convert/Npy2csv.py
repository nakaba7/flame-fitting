import numpy as np
import csv
import argparse
"""
npyファイルをcsvファイルに変換するスクリプト。
Usage:
    python Npy2csv.py -i input.npy -o output.csv
Args:   
    -i: 入力npyファイル
    -o: 出力csvファイル
"""

def npy2csv(npy_file, csv_file):
    # npyファイルを読み込む
    data = np.load(npy_file)

    # CSVファイルに書き出す
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print(f"Saved {csv_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='input npy file')
    parser.add_argument('-o', type=str, help='output csv file')
    args = parser.parse_args()
    npy2csv(args.i, args.o)