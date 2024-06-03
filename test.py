import os
import pandas as pd 
import numpy as np

def get_valid_indices(expr_dir, pose_dir):
    expr_files = sorted([f for f in os.listdir(expr_dir) if f.startswith('test') and f.endswith('_expr.npy')])
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.startswith('test') and f.endswith('_pose.npy')])
    # 存在するインデックスを取得
    valid_indices = set(int(f[4:9]) for f in expr_files) & set(int(f[4:9]) for f in pose_files)
    return sorted(valid_indices)

def filter_csv_by_indices(input_csv, output_csv, valid_indices):
    # CSVファイルを読み込む
    csv_data = pd.read_csv(input_csv, header=None)
    csv_data = csv_data.values
    new_csv_data = []
    for i in range(len(csv_data)):
        if i in valid_indices:
            new_csv_data.append(csv_data[i])
        else:
            print(f"Skipping index {i}")
    np.savetxt(output_csv, new_csv_data, delimiter=',', fmt='%s')

expr_dir = 'output_params/Nakabayashi/expr'  # 例: 'C:/path/to/expr_dir'
pose_dir = 'output_params/Nakabayashi/pose'  # 例: 'C:/path/to/pose_dir'
valid_indices = get_valid_indices(expr_dir, pose_dir)

# CSVファイルのフィルタリングと保存
input_csv = 'sensor_values/Nakabayashi/sensor_data_test.csv'
output_csv ='sensor_values/Nakabayashi/sensor_data_test_filtered.csv'
filter_csv_by_indices(input_csv, output_csv, valid_indices)