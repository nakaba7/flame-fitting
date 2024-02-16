import numpy as np
name = "Nakabayashi"
# 画像1, 2における特徴点の座標
points1 = np.load(f"../data/scan_lmks.npy")
#points2 = np.load(f"points_b_{name}.npy")

# 三次元座標を計算

for point1, point2 in zip(points1, points2):
    print("point1: \n",point1)
    print("point2: \n",point2)


print("points1: \n",points1*100)
print("points1.shape: \n",points1.shape)
