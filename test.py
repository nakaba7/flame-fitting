import numpy as np

def load_and_print_npy(file_path):
    data = np.load(file_path)
    print("Data loaded from file:")
    print(data)

#file_path = 'inputs_cache_200000_normalized.npy'  # 実際のファイルパスに置き換えてください
file_path = 'output_landmark/2d/lmk_0.npy'  # 実際のファイルパスに置き換えてください
load_and_print_npy(file_path)
