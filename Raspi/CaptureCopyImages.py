import socket
import cv2
from picamera2 import Picamera2
import datetime
import shutil
import os

def capture_image(image_index):
    image = camera.capture_array()
    imgpath = f"images/test{image_index}.jpg"
    cv2.imwrite(imgpath, image)
    print(imgpath, "saved!")
    return imgpath  # 保存された画像のパスを返す

def copy_images_to_destination(images, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for image_path in images:
        shutil.copy(image_path, destination_folder)
    print(f"All images have been copied to {destination_folder}.")

# ホストとポートを設定
HOST = '0.0.0.0'  # すべてのネットワークインターフェイスでリッスン
PORT = 46361      # ポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT} for signals to capture images.")
    camera = Picamera2()
    camera.resolution = (640, 480)
    camera.framerate = 15
    camera.start()

    captured_images = []  # 撮影された画像のパスを格納するリスト
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            image_index = 0
            while True:
                data = conn.recv(1024).decode()
                if data == 'q' or not data:
                    camera.stop()
                    print("Connection closed.")
                    # 撮影終了後、画像を目的のフォルダにコピー
                    destination_folder = "/path/to/destination_folder"
                    copy_images_to_destination(captured_images, destination_folder)
                    # 画像コピー完了後、終了合図として'q'を通信相手に送信
                    conn.sendall(b'q')
                    conn.close()  # コネクションを閉じる
                    exit()
                # データ受信後、画像を撮影
                print(datetime.datetime.now())
                imgpath = capture_image(image_index)
                captured_images.append(imgpath)  # 撮影された画像のパスをリストに追加
                image_index += 1
