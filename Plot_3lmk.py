import matplotlib.pyplot as plt
import numpy as np

def plot_3lmk(landmark_2d, landmark_3d, target_3d):
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
