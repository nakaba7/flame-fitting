import os
import numpy as np
import cv2
from cv2 import aruco
from CaptureArucoMarker import capture_window_screenshot
import socket
import subprocess
import time

#Unityのパススルーシーンで撮影した画像からリアルタイムでマーカーの情報を取得し、UnityへArucoマーカーの並進ベクトル、回転角(オイラー角))を渡す。
#送信データはUnityで使えるように、左手系仕様 → 右手系仕様にしてから送信するようになっている。

#Meta Quest Proのどちらのレンズを撮影カメラとするか
capture_eye = "left"
#capture_eye = "right"

def getArucoMiddlePoint(corners):
    #Arucoマーカーの中央の座標を返す
    upper_midpoint = (corners[0][0][0]+corners[0][0][1])/2
    lower_midpoint = (corners[0][0][2]+corners[0][0][3])/2
    return (upper_midpoint+lower_midpoint)/2

def main():
    #スクショ保存ディレクトリが存在しなければ生成
    if os.path.isdir(os.getcwd()+"\\ArucoForRealTime") == False:
        os.mkdir(os.getcwd()+"\\ArucoForRealTime")

    # scrcpy.exeのパスと起動オプションを設定
    scrcpy_path = "./scrcpy-win64-v2.1.1/scrcpy.exe"

    #アスペクト比 = a : b、開始位置(x,y) の場合、'a:b:x:y'と指定する。
    left_eye_capture = '1750:1900:80:0'
    right_eye_capture = '1750:1900:1850:0'
    if capture_eye=="left":
        scrcpy_options = ['--always-on-top', '--crop', left_eye_capture]
        angle = -15
    elif capture_eye=="right":
        scrcpy_options = ['--always-on-top', '--crop', right_eye_capture]
        angle = 15
    
    # subprocessでscrcpy.exeを起動
    scrcpy_process = subprocess.Popen([scrcpy_path] + scrcpy_options)

    # マーカーサイズ
    marker_length = 0.060 # [m]
    # マーカーに描写する軸の長さ
    #draw_pole_length = marker_length/2 
    # マーカーの辞書選択
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    #CameraCalibration.py にて得たカメラの内部パラメータ
    camera_matrix = np.load("mtx_"+capture_eye+"_scrcpy.npy")
    distortion_coeff = np.load("dist_"+capture_eye+"_scrcpy.npy")
    
    #UDP通信用設定
    HOST = '127.0.0.1'
    PORT = 50007
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    #送信用外部パラメータの初期値
    marker_position_rotation_str = "0,0,0,0,0,0"
    
    screenshot_name = "ArucoForRealTime/"+f"screenshot.png"
    
    print("Waiting for scrcpy...")
    time.sleep(2)
    print("Capture Start!")
    
    try:
        # scrcpyプロセスが終了するまでループ
        while scrcpy_process.poll() is None:
            #scrcpyのウィンドウをスクリーンショットし，画像として保存
            capture_window_screenshot("Quest Pro", screenshot_name)
            img = cv2.imread(screenshot_name)    
            height, width = img.shape[:2]
            center = (width/2, height/2)
            #アフィン変換行列を作成する
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
            #アフィン変換行列を画像に適用する
            img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
            corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
            if len(corners)>0:
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion_coeff)
                #右手系の回転ベクトル(u,v,w)は左手系だと(-u,-v,-w)となる???(-u,v,-w)と考えたが違った．
                rvec[0][0][0] = -rvec[0][0][0]
                rvec[0][0][1] = -rvec[0][0][1]
                rvec[0][0][2] = -rvec[0][0][2]
                #aruco.drawAxis(img, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)
                #位置座標は右手系 → 左手系で (x,y,z) → (x,-y,z)
                tvec[0][0][1] = -tvec[0][0][1]
                #回転ベクトルを回転行列へ変換する
                rvec_matrix = cv2.Rodrigues(rvec[0][0])
                # rodoriguesから抜き出し
                rvec_matrix = rvec_matrix[0]
                # 並進ベクトルの転置
                transpose_tvec = tvec[0][0][np.newaxis, :].T
                # 透視投影行列を作成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
                euler_angle = np.squeeze(euler_angle)
                #送信のため，位置座標と回転角をstring型にして","で繋げる．(x,y,z,roll(x軸周り回転),pitch(y軸周り回転),yaw(z軸周り回転))
                marker_position_rotation_str = str(tvec[0][0][0])+','+str(tvec[0][0][1])+','+str(tvec[0][0][2])+','+str(euler_angle[0])+','+str(euler_angle[1])+','+str(euler_angle[2])
                #Unityへ送信
                client.sendto(marker_position_rotation_str.encode('utf-8'),(HOST,PORT))
                print(marker_position_rotation_str)
            else:
                client.sendto(marker_position_rotation_str.encode('utf-8'),(HOST,PORT))
                print("No Marker")
            cv2.imwrite("ArucoForRealTime/"+f"screenshot_rotate.png", img)
    except KeyboardInterrupt:
        # Ctrl+Cが押された場合はプロセスを終了
        scrcpy_process.terminate()


if __name__ == "__main__":
    main()
