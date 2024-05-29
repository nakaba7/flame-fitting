import numpy as np
import os
from tqdm import tqdm  # tqdmをインポート

def lmk3d_2_2d(lmk_3d):
    """
    3次元ランドマークを2次元ランドマークに変換する関数.
    """
    # 3次元目の最初の2要素を保持
    lmk_2d = lmk_3d[:, :2]
    return lmk_2d

