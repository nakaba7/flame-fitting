import cv2
from picamera2 import Picamera2
import time

def capture_image(camera0, camera1, image_index):
    # カメラ0から画像を取得
    image0 = camera0.capture_array()
    imgpath0 = f"test{image_index}_0.jpg"
    cv2.imwrite(imgpath0, image0)

    # カメラ1から画像を取得
    image1 = camera1.capture_array()
    imgpath1 = f"test{image_index}_1.jpg"
    cv2.imwrite(imgpath1, image1)

    print(imgpath0, "and", imgpath1, "saved!")

camera0 = Picamera2(0)
camera1 = Picamera2(1)

# それぞれのカメラに対して設定を行い、起動
config0 = camera0.create_preview_configuration(main={"size": (1640, 1232)})
camera0.configure(config0)
camera0.start()

config1 = camera1.create_preview_configuration(main={"size": (1640, 1232)})
camera1.configure(config1)
camera1.start()

image_index = 0

#1秒ごとに画像を取得
while True:
    capture_image(camera0, camera1, image_index)
    time.sleep(1)
    image_index += 1

camera0.stop()
camera1.stop()
