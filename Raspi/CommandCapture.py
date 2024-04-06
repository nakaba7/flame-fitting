import socket
import cv2
from picamera2 import Picamera2
import datetime
from imagerotate import rotate_images

def capture_image(camera1, camera2, image_index):
    # カメラ1から画像を取得
    image1 = camera1.capture_array()
    imgpath1 = f"images/test{image_index}_0.jpg"
    cv2.imwrite(imgpath1, image1)

    # カメラ2から画像を取得
    image2 = camera2.capture_array()
    imgpath2 = f"images/test{image_index}_1.jpg"
    cv2.imwrite(imgpath2, image2)

    print(imgpath1, "and", imgpath2, "saved!")

# ホストとポートを設定
HOST = '0.0.0.0'  # すべてのネットワークインターフェイスでリッスン
PORT = 46361      # ポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Listening on {HOST}:{PORT} for signals to capture images.")

    # 2つのカメラインスタンスを作成
    camera1 = Picamera2(0)
    camera2 = Picamera2(1)

    # それぞれのカメラに対して設定を行い、起動
    config1 = camera1.create_preview_configuration(main={"size": (1640, 1232)})
    camera1.configure(config1)
    camera1.start()

    config2 = camera2.create_preview_configuration(main={"size": (1640, 1232)})
    camera2.configure(config2)
    camera2.start()

    image_index = 0
    try:
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if data == b'q' or not data:
                        raise Exception("Quit signal received")
                    # データ受信後、画像を撮影
                    print(datetime.datetime.now())
                    capture_image(camera1, camera2, image_index)
                    image_index += 1
    except Exception as e:
        print(e)
    finally:
        camera1.stop()
        camera2.stop()
        s.close()
        # 撮影した画像を90°回転
        print("Rotate Images...")
        rotate_images("images")
        print("Connection closed.")
