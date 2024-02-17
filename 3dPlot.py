import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
npyファイルを読み込んで3Dプロットするスクリプト
"""

data = np.load("output_landmark/angry_3d.npy")
print(data)
data[:, 2] = -data[:, 2]
#print(data)

# プロットを初期化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z のデータを抽出
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

# 散布図としてプロット
scatter = ax.scatter(X, Y, Z)

# ラベルを設定
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 各点の近くにインデックス番号を表示
for i in range(len(X)):
    ax.text(X[i], Y[i], Z[i], f'{i}', color='blue', fontsize=9)

# 表示
plt.show()
