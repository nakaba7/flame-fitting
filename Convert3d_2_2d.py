import numpy as np
import os
from tqdm import tqdm  # tqdmをインポート

# 入力ディレクトリと出力ディレクトリの設定
input_dir = 'output_landmark/3d/test'
output_dir = 'output_landmark/2d/test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"{output_dir} created")

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 入力ディレクトリ内のすべてのファイル名を取得
files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

# tqdmを使って進行状況バーを表示
for file_name in tqdm(files, desc="Processing files"):
    # ファイルをロード
    file_path = os.path.join(input_dir, file_name)
    data = np.load(file_path)
    
    # 3次元目の最初の2要素を保持
    data_reduced = data[:, :2]
    
    # 新しいファイル名を設定し、データを保存
    output_file_path = os.path.join(output_dir, file_name)
    np.save(output_file_path, data_reduced)
