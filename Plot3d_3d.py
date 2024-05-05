import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def plot_landmarks3d_double(landmarks_3d_1, landmarks_3d_2):
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

