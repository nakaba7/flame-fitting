import socket
import datetime
"""
Raspberry Piにコマンドを送信するスクリプト
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST = "192.168.11.57"  # Raspberry Piのホスト名またはIPアドレスを設定
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")

    while True:
        cmd = input()  # ユーザーの入力を待つ
        if cmd.lower() == 'q':
            s.sendall(b'q')
            break  
        # Enterが押されたらRaspberry Piにシグナルを送信
        s.sendall(b'c')
        print(datetime.datetime.now())
        print("Signal sent to Raspberry Pi to capture an image.")


