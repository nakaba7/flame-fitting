import numpy as np
import sys
import os

# 親ディレクトリをパスに追加
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# moduleA.pyをインポート
from lmk_plot import plot_2d

# 関数を呼び出す
plot_2d(np.load("../FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy"))
