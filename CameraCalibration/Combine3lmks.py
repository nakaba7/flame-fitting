import numpy as np
import matplotlib.pyplot as plt

"""
左目, 右目, 口元のランドマークを結合してプロットするスクリプト. 各ランドマークはカメラの内部パラメータや外部パラメータを使わず, そのまま繋なげているだけ.
"""

def plot_contour(origin, width, height, color, label):
    # 画像の輪郭を描画し、周囲の長さを計算して表示する関数
    x_coords = [origin[0], origin[0] + width, origin[0] + width, origin[0], origin[0]]
    y_coords = [origin[1], origin[1], origin[1] + height, origin[1] + height, origin[1]]
    plt.plot(x_coords, y_coords, color + '-', label=label)
    
    # 各辺の長さを表示
    edge_lengths = [width, height, width, height]
    midpoints = [
        ((x_coords[i] + x_coords[i + 1]) / 2, (y_coords[i] + y_coords[i + 1]) / 2)
        for i in range(4)
    ]
    for (mx, my), length in zip(midpoints, edge_lengths):
        plt.text(mx, my, f'{length}', fontsize=10, ha='center', va='center', color=color)
    
    contour_length = 2 * (width + height)
    return contour_length

def lmk_sort(all_data):
    #顔のランドマークをflame-fittingの順番に並び替える
    new_lmks = all_data.copy()
    new_lmks[5:10] = all_data[42:47]#左眉毛
    new_lmks[10:14] = all_data[47:51]#鼻上部
    new_lmks[14:19] = all_data[11:16]#鼻下部   
    new_lmks[19:25] = all_data[5:11]#右目
    new_lmks[25:31] = all_data[36:42]#左目
    new_lmks[31:51] = all_data[16:36]#口
    return new_lmks

def combine3lmks(lmk_right, lmk_mouth, lmk_left):
    # 各ファイルの原点を設定
    origin_right = np.array([0, 0])
    origin_left = np.array([1232, 0])
    origin_mouth = np.array([412, 1640])

    # 各データに原点の位置を加算し、y座標を反転して画像座標系に合わせる
    lmk_right_transformed = lmk_right + origin_right
    lmk_mouth_transformed = lmk_mouth + origin_mouth
    lmk_left_transformed = lmk_left + origin_left

    # すべてのデータを結合
    all_data = np.concatenate((lmk_right_transformed, lmk_mouth_transformed, lmk_left_transformed), axis=0)

    # 5番目の点と39番目の点の中点を計算
    midpoint = (all_data[5] + all_data[39]) / 2

    # 中点から13番の方向へ一定間隔で3つの点を追加
    direction = all_data[13] - midpoint
    interval = np.linalg.norm(direction) / 4  # 一定間隔を計算
    unit_direction = direction / np.linalg.norm(direction)

    new_points = [midpoint + unit_direction * interval * (i) for i in range(4)]
    new_points = np.array(new_points)

    # 新しい点をall_dataに追加
    all_data = np.concatenate((all_data, new_points), axis=0)
    all_data = lmk_sort(all_data)

    # 中央に移動させる
    center = np.mean(all_data, axis=0)
    all_data = all_data - center

    # スケーリング
    max_range = np.max(np.ptp(all_data, axis=0))  # ptp: peak to peak (range)
    scale_factor = 0.1 / max_range  # スケーリングファクターを計算
    all_data_scaled = all_data * scale_factor
    all_data_scaled[:,1] *= -1

    return all_data_scaled

if __name__ == '__main__':
    # 各.npyファイルからデータを読み込み
    lmk_left = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_0_annotated.npy")
    lmk_mouth = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_annotated.npy")
    lmk_right = np.load("../FaceData/FaceImages_Annotated/NPYs/test6_1_annotated.npy")

    all_data = combine3lmks(lmk_right, lmk_mouth, lmk_left)

    # プロット
    plt.figure(figsize=(10, 10))
    plt.scatter(all_data[:, 0], all_data[:, 1], s=5, c='r')

    # インデックスと座標を表示
    for idx, (x, y) in enumerate(all_data):
        plt.text(x, y, str(idx), fontsize=8, ha='right')
        #print(f"Index: {idx}, Coordinates: ({x}, {y})")

    """
    # lmk_right の輪郭
    right_length = plot_contour(origin_right, 1232, 1640, 'b', 'Right Contour')

    # lmk_mouth の輪郭
    mouth_length = plot_contour(origin_mouth, 1640, 1232, 'g', 'Mouth Contour')

    # lmk_left の輪郭
    left_length = plot_contour(origin_left, 1232, 1640, 'b', 'Left Contour')
    """
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # y軸を反転
    plt.axis('equal')  # x軸とy軸のスケールを同じにする
    plt.title('Merged 2D Face Feature Points')
    plt.grid(True)
    #plt.legend()
    plt.show()