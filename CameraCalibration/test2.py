import numpy as np
import matplotlib.pyplot as plt

def rotate_around_y_axis(points, angle_degrees):
    # 角度をラジアンに変換
    angle_radians = np.radians(angle_degrees)
    
    # y軸周りの回転行列
    rotation_matrix_y = np.array([
        [np.cos(angle_radians), 0, np.sin(angle_radians)],
        [0, 1, 0],
        [-np.sin(angle_radians), 0, np.cos(angle_radians)]
    ])
    
    # 2次元の特徴点を3次元に拡張 (z=0)
    points_3d = np.array([[x, y, 0] for x, y in points])
    
    # 回転行列を適用して新しい座標を計算
    rotated_points_3d = np.dot(points_3d, rotation_matrix_y.T)
    
    # z軸方向からの投影 (x, zを使う)
    projected_points = [(x, z) for x, y, z in rotated_points_3d]
    
    return projected_points

# 例として特徴点のリストを定義
points = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")

# 60°回転させた方向から見た特徴点を計算
rotated_points = rotate_around_y_axis(points, 60)

# 元の特徴点と回転後の特徴点をプロット
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 元の特徴点をプロット
original_points_x = [x for x, y in points]
original_points_y = [y for x, y in points]
ax1.scatter(original_points_x, original_points_y, color='blue', label='Original Points')

# 回転後の特徴点をプロット
rotated_points_x = [x for x, y in rotated_points]
rotated_points_y = [y for x, y in rotated_points]
ax2.scatter(rotated_points_x, rotated_points_y, color='red', label='Rotated Points')

# プロットの設定
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Original 2D Feature Points')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Feature Points Rotated around Y-axis by 60 Degrees')
ax2.legend()
ax2.grid(True)

# グラフを表示
plt.show()