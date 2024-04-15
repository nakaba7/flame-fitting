import socket
import datetime

"""
2つのRaspberry Piにコマンドを送信するスクリプト
Wi-fi経由で行う
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.100.34"  #Raspberry Pi 5(両目カメラ)
HOST_B = "192.168.100.37"  #Raspberry Pi 3(口周辺カメラ)
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
        sa.connect((HOST_A, PORT))
        sb.connect((HOST_B, PORT))
        print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")
        
        while True:
            cmd = input()  # ユーザーの入力を待つ
            if cmd.lower() == 'q':
                sa.sendall(b'q')
                sb.sendall(b'q')
                exit()

            # Enterが押されたらRaspberry Piにシグナルを送信
            print(datetime.datetime.now())
            
            sa.sendall(b'c')
            sb.sendall(b'c')
            print("Signal sent to Raspberry Pi to capture an image.")


