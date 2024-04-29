import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

"""
npyファイルに保存された2D, 3Dランドマークをプロットするスクリプト
qを押すとウィンドウが閉じ, 次の画像のプロットに移る. 機械学習により2d→3dが上手くいっているかの確認に使う.
"""

def plot_landmarks(landmarks_2d, landmarks_3d):
    fig = plt.figure(figsize=(14, 7))
    
    # 2Dランドマークのプロット
    ax1 = fig.add_subplot(121)
    ax1.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1])
    for i, txt in enumerate(range(landmarks_2d.shape[0])):
        ax1.annotate(txt, (landmarks_2d[i, 0], landmarks_2d[i, 1]))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Landmarks')
    
    # 3Dランドマークのプロット
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])
    for i, txt in enumerate(range(landmarks_3d.shape[0])):
        ax2.text(landmarks_3d[i, 0], landmarks_3d[i, 1], landmarks_3d[i, 2], txt)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Landmarks')

    # キー入力イベントを処理する関数
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # キープレスイベントに反応するように設定
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    files_2d = sorted(glob.glob('output_landmark/2d/*.npy'))
    files_3d = sorted(glob.glob('output_landmark/3d/*.npy'))

    for f2d, f3d in zip(files_2d, files_3d):
        landmarks_2d = np.load(f2d)
        landmarks_3d = np.load(f3d)
        print("2d:",landmarks_2d)
        print("3d:",landmarks_3d)
        if landmarks_3d.shape[0] == 1:
            landmarks_3d = np.squeeze(landmarks_3d)
        landmarks_2d[:, 1] = -landmarks_2d[:, 1]# 画像座標系とプロット座標系を合わせるためにy座標を反転
        landmarks_3d[:, 1] = -landmarks_3d[:, 1]
        print("loaded", f2d, f3d)
        plot_landmarks(landmarks_2d, landmarks_3d)

if __name__ == '__main__':
    main()
