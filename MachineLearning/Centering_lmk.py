import numpy as np
from tqdm import tqdm

def centering_lmk(npy_file_path):
    """
    2次元または3次元のランドマークを原点に合わせ, 上書き保存する関数.
    Args:
        npy_file_path: ランドマークのnpyファイルパス
    """
    lmk_nparray = np.load(npy_file_path)
    print(f"Centering {npy_file_path}")
    for i in tqdm(range(lmk_nparray.shape[0])):
        reference_point = lmk_nparray[i, 16]
        lmk_nparray[i] -= reference_point
    print("Saving centered landmarks...\n")
    np.save(npy_file_path, lmk_nparray)

if __name__ == "__main__":
    input_npy = "inputs_cache_200000.npy"
    target_npy = "targets_cache_200000.npy"
    centering_lmk(input_npy)
    centering_lmk(target_npy)

