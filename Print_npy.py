import numpy as np

# 画像1, 2における特徴点の座標
points1 = np.load("output_landmark/estimated_3d/000133.npy")
#points2 = np.load(f"points_b_{name}.npy")

# 三次元座標を計算

print("points1: \n",points1)
print("points1.shape: \n",points1.shape)
