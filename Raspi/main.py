import socket
import datetime
import time
import argparse
import sys
import os
import csv

# 親ディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)

from AffectiveHMD.Quest2_affectiveHMD import setup, write_csv
"""
ラズパイ2つから画像データ, Arduinoからフォトリフレクタの値を取得するスクリプト.
usage:
    python main.py -n IMAGENUM -p PARTICIPANTNAME
args:
    -n: 画像をキャプチャする回数を指定
    -p: 参加者名を指定
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.100.34"  #Raspberry Pi 5(両目カメラ)
HOST_B = "192.168.100.45"  #Raspberry Pi 3(口周辺カメラ)
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号
sleeptime = 0.2

def wait_for_ack(sock):
    while True:
        data = sock.recv(1024)
        if data == b'captured':
            break

def main(args):
    image_num = args.n
    participant_name = args.p
    output_dir = f'../sensor_values/{participant_name}'
    ser, csv_file_path = setup(output_dir=output_dir, port='COM11', baudrate=115200)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
            with open(csv_file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                print(f"Opened csv file: {csv_file_path}")
                sa.connect((HOST_A, PORT))
                sb.connect((HOST_B, PORT))
                print("Connected to Raspberry Pi.")
                i=0
                print("Start after 3 seconds...")
                for j in range(3):
                    print(3-j)
                    time.sleep(1)
                print("Start!")

                while True:
                    # 画像をキャプチャする上限回数に達したら終了
                    if i==image_num:
                        sa.sendall(b'q')
                        sb.sendall(b'q')
                        ser.close()
                        print(f"{image_num} images captured. Quitting...")
                        break

                    sa.sendall(b'c')
                    sb.sendall(b'c')
                    wait_for_ack(sa)  
                    wait_for_ack(sb)
                    ser.reset_input_buffer()  # シリアルバッファをクリア
                    ser.write(b'c')#Arduinoに合図を送る
                    write_csv(ser, csvwriter)#フォトリフレクタの値を取得してcsvに追記
                    print(datetime.datetime.now())
                    print(f"Image {i+1} captured.")
                    i+=1
                    time.sleep(sleeptime)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=100 , type=int,  help='Enter the image num to capture.')
    parser.add_argument('-p', default='hoge', type=str,  help='Enter the participant name.')
    main(parser.parse_args())