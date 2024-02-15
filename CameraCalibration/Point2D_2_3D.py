import numpy as np
import cv2
import argparse

def triangulate_points(P1, P2, points1, points2):
    # P1, P2: カメラa, bの射影行列
    # points1: 画像1における特徴点の座標のリスト
    # points2: 画像2における特徴点の座標のリスト

    # 点を同次座標系に変換
    points1_hom = cv2.convertPointsToHomogeneous(points1)
    points2_hom = cv2.convertPointsToHomogeneous(points2)

    # 三角測量を実行
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)

    # 同次座標から非同次座標へ変換
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)

    return points_3d

def main(args):
    name = args.n
    """
    # カメラa, bの内部パラメータと歪み係数
    mtx_a = np.load("ChessBoard_a_mtx.npy")
    dist_a = np.load("ChessBoard_a_dist.npy")
    mtx_b = np.load("ChessBoard_b_mtx.npy")
    dist_b = np.load("ChessBoard_b_dist.npy")

    # カメラa, bの外部パラメータ
    R = np.load("R.npy")
    T = np.load("T.npy")
    """
    # カメラa, bの射影行列
    P1 = np.load("P1.npy")
    P2 = np.load("P2.npy")
    #↑計算で出すか，キャリブレーションで出したものを使うか

    # 画像1, 2における特徴点の座標
    points1 = np.load(f"points_a_{name}.npy")
    points2 = np.load(f"points_b_{name}.npy")

    # 三次元座標を計算
    for point1, point2 in zip(points1, points2):
        point_3d = triangulate_points(P1, P2, point1, point2)
        print(point1,point2)
        print("↓")
        print(point_3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, help='Participant name')
    main(parser.parse_args())