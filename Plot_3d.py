import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_landmarks(landmarks_3d):
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
