import socket
import datetime
import time
import argparse
"""
2つのRaspberry Piに1秒ごとにコマンドを送信するスクリプト
usage:
    python CaptureChessCommand2Raspi.py -n IMAGENUM -c both
    python CaptureChessCommand2Raspi.py -n IMAGENUM -c eyes
    python CaptureChessCommand2Raspi.py -n IMAGENUM -c mouth
args:
    -n: 画像をキャプチャする回数を指定
    -c: どちらのカメラを使用するかを指定。both, eyes, mouthのいずれかを選択
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.100.34"  #Raspberry Pi 5(両目カメラ)
HOST_B = "192.168.100.45"  #Raspberry Pi 3(口周辺カメラ)
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号
sleeptime = 1

def wait_for_ack(sock):
    while True:
        data = sock.recv(1024)
        if data == b'captured':
            print("Image captured.")
            break

def main(args):
    image_num = args.n
    if args.c == 'both':
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
                sa.connect((HOST_A, PORT))
                sb.connect((HOST_B, PORT))
                print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")
                i=0
                print("Start after 3 seconds...")
                for j in range(3):
                    print(3-j)
                    time.sleep(1)
                print("Start!")
                while True:
                    if i==image_num:
                        sa.sendall(b'q')
                        sb.sendall(b'q')
                        print(f"{image_num} images captured. Quitting...")
                        break
                    sa.sendall(b'c')
                    sb.sendall(b'c')
                    wait_for_ack(sa)  # 合図を待つ
                    wait_for_ack(sb)  # 合図を待つ
                    print(datetime.datetime.now())
                    print(f"Image {i+1} captured.")
                    i+=1
                    time.sleep(sleeptime)
    elif args.c == 'eyes':
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
            sa.connect((HOST_A, PORT))
            print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")
            i=0
            print("Start after 3 seconds...")
            for j in range(3):
                print(3-j)
                time.sleep(1)
            print("Start!")
            while True:
                if i==image_num:
                    sa.sendall(b'q')
                    print(f"{image_num} images captured. Quitting...")
                    break
                sa.sendall(b'c')
                #wait_for_ack(sa)  # 合図を待つ
                print(datetime.datetime.now())
                print(f"Image {i+1} captured.")
                i+=1
                time.sleep(sleeptime)
    elif args.c == 'mouth':
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
            sb.connect((HOST_B, PORT))
            print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")
            i=0
            print("Start after 3 seconds...")
            for j in range(3):
                print(3-j)
                time.sleep(1)
            print("Start!")
            while True:
                if i==image_num:
                    sb.sendall(b'q')
                    print(f"{image_num} images captured. Quitting...")
                    break
                sb.sendall(b'c')
                wait_for_ack(sb)  # 合図を待つ
                print(datetime.datetime.now())
                print(f"Image {i+1} captured.")
                i+=1
                time.sleep(sleeptime)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=50 , type=int,  help='Enter the image num to capture.')
    parser.add_argument('-c', default='both', type=str,  help='whether use both or eyes or mouth. Choose both, eyes or mouth.')
    main(parser.parse_args())