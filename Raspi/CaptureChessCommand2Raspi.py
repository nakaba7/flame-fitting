import socket
import datetime
import time

"""
2つのRaspberry Piに1秒ごとにコマンドを送信するスクリプト
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.100.34"  #Raspberry Pi 5(両目カメラ)
HOST_B = "192.168.100.37"  #Raspberry Pi 3(口周辺カメラ)
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号

image_num = 50

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
        sa.connect((HOST_A, PORT))
        sb.connect((HOST_B, PORT))
        print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")
        i=0
        print("Start after 3 seconds...")
        time.sleep(3)
        print("Start!")
        while True:
            if i==image_num:
                sa.sendall(b'q')
                sb.sendall(b'q')
                print("30 images captured. Quitting...")
                break
            sa.sendall(b'c')
            sb.sendall(b'c')
            print(datetime.datetime.now())
            print(f"Image {i+1} captured.")
            i+=1
            time.sleep(1)



