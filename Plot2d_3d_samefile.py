import numpy as np
import matplotlib.pyplot as plt
import glob

"""
3次元ランドマークを2次元座標と3次元座標にプロットするスクリプト.
Plot2d_3d.pyと違う点は, 別ファイルの2次元ランドマークを使わない点. 3d→2dが上手くいっているかの確認に使う.
"""

def plot_landmarks(landmarks_3d):
    fig = plt.figure(figsize=(12, 6))
    
    # 2Dランドマークのプロット
    ax0 = fig.add_subplot(121)  # 1行2列の1番目
    ax0.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1])
    for i, txt in enumerate(range(landmarks_3d.shape[0])):
        ax0.text(landmarks_3d[i, 0], landmarks_3d[i, 1], txt, fontsize=9)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_title('2D Landmarks')

    # 3Dランドマークのプロット
    ax1 = fig.add_subplot(122, projection='3d')  # 1行2列の2番目
    ax1.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])
    for i, txt in enumerate(range(landmarks_3d.shape[0])):
        ax1.text(landmarks_3d[i, 0], landmarks_3d[i, 1], landmarks_3d[i, 2], txt)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Landmarks')

    # キー入力イベントを処理する関数
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # キープレスイベントに反応するように設定
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

def main():
    files_3d = sorted(glob.glob('output_landmark/3d/*.npy'))

    for f3d in files_3d:
        landmarks_3d = np.load(f3d)
        if landmarks_3d.shape[0] == 1:
            landmarks_3d = np.squeeze(landmarks_3d)
        landmarks_3d[:, 1] = -landmarks_3d[:, 1]  # Y軸反転
        print("loaded", f3d)
        plot_landmarks(landmarks_3d)

if __name__ == '__main__':
    main()
