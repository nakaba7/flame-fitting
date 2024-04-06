import socket
import cv2
from picamera2 import Picamera2
import datetime
from imagerotate import rotate_images

def capture_image(image_index):
    image = camera.capture_array()
    imgpath = f"images/test{image_index}.jpg"
    cv2.imwrite(imgpath, image)
    print(imgpath, "saved!")

# ホストとポートを設定
HOST = '0.0.0.0'  # すべてのネットワークインターフェイスでリッスン
PORT = 46361      # ポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT} for signals to capture images.")
    camera = Picamera2()

    # カメラの解像度設定を更新
    config = camera.create_preview_configuration(main={"size": (1640, 1232)})
    camera.configure(config)

    camera.start()

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            image_index = 0
            while True:
                data = conn.recv(1024)
                if data=='q' or not data:
                    camera.stop()
                    s.close()
                    # 撮影した画像を90°回転
                    print("Rotate Images...")
                    rotate_images("images", 90)
                    print("Connection closed.")
                    exit()  
                # データ受信後、画像を撮影
                print(datetime.datetime.now())
                capture_image(image_index)
                image_index += 1
