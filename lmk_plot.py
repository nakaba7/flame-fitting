import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_3d(landmarks_3d):
    """
    3次元ランドマークをプロットする関数.
    """
    fig = plt.figure(figsize=(12, 6))
    
    # 3Dランドマークのプロット
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])
    for i, txt in enumerate(range(landmarks_3d.shape[0])):
        ax1.text(landmarks_3d[i, 0], landmarks_3d[i, 1], landmarks_3d[i, 2], txt)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Landmarks')

    # 軸の範囲を計算し設定
    all_data = np.vstack([landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2]])
    max_range = np.array([all_data.max() - all_data.min()]).max() / 2.0
    mid_x = (max(landmarks_3d[:, 0]) + min(landmarks_3d[:, 0])) * 0.5
    mid_y = (max(landmarks_3d[:, 1]) + min(landmarks_3d[:, 1])) * 0.5
    mid_z = (max(landmarks_3d[:, 2]) + min(landmarks_3d[:, 2])) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # キー入力イベントを処理する関数
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # キープレスイベントに反応するように設定
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()



def plot_3lmk(landmark_2d, landmark_3d, target_3d):
    """
    左から, 2次元ランドマーク, 予測された3次元ランドマーク, 正解の3次元ランドマークをプロットする関数.
    """
    fig = plt.figure(figsize=(21, 7))
    
    # 2D Landmarks plot
    ax1 = fig.add_subplot(131)  # 通常の2次元プロット
    ax1.scatter(landmark_2d[:, 0], landmark_2d[:, 1], c='r', label='2D Landmarks')
    for i, txt in enumerate(landmark_2d):
        ax1.annotate(str(i), (landmark_2d[i, 0], landmark_2d[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
    ax1.set_title('2D Landmarks')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # Predicted 3D landmarks
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(landmark_3d[:, 0], landmark_3d[:, 1], landmark_3d[:, 2], c='b', label='Predicted 3D Landmarks')
    for i, txt in enumerate(landmark_3d):
        ax2.text(landmark_3d[i, 0], landmark_3d[i, 1], landmark_3d[i, 2], str(i), color='black')
    ax2.set_title('Predicted 3D Landmarks')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    # Ground Truth 3D landmarks
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2], c='g', label='Ground Truth 3D Landmarks')
    for i, txt in enumerate(target_3d):
        ax3.text(target_3d[i, 0], target_3d[i, 1], target_3d[i, 2], str(i), color='black')
    ax3.set_title('Ground Truth 3D Landmarks')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()

    plt.show()


def plot_2d_3d(landmarks_3d):
    """
    3次元ランドマークを2次元座標と3次元座標にプロットするスクリプト.
    """
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


def plot_2d_3d_compare(landmarks_2d, landmarks_3d):
    """
    npyファイルに保存された2D, 3Dランドマークをプロットするスクリプト
    qを押すとウィンドウが閉じ, 次の画像のプロットに移る. 機械学習により2d→3dが上手くいっているかの確認に使う.
    """
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


def plot_3d_3d_compare(landmarks_3d_1, landmarks_3d_2):
    """
    2つの3次元ランドマークを比較する関数.
    """
    fig = plt.figure(figsize=(14, 7))
    
    # First set of 3D landmarks
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(landmarks_3d_1[:, 0], landmarks_3d_1[:, 1], landmarks_3d_1[:, 2], color='blue')
    for i, point in enumerate(landmarks_3d_1):
        ax1.text(point[0], point[1], point[2], str(i), color='red', fontsize=9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Landmarks Set 1')
    
    # Second set of 3D landmarks
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(landmarks_3d_2[:, 0], landmarks_3d_2[:, 1], landmarks_3d_2[:, 2], color='blue')
    for i, point in enumerate(landmarks_3d_2):
        ax2.text(point[0], point[1], point[2], str(i), color='red', fontsize=9)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Landmarks Set 2')
    
    # キー入力イベントを処理する関数
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # キープレスイベントに反応するように設定
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
