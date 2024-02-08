import socket

"""
Raspberry Piにコマンドを送信するスクリプト
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST = 'raspberrypi.local'  # Raspberry Piのホスト名またはIPアドレスを設定
PORT = 12345                # Raspberry Pi側のスクリプトで使用しているポート番号

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")

    while True:
        cmd = input()  # ユーザーの入力を待つ
        if cmd.lower() == 'q':
            break  # 'q'が入力されたら終了

        # Enterが押されたらRaspberry Piにシグナルを送信
        s.sendall(b'Capture')
        print("Signal sent to Raspberry Pi to capture an image.")
