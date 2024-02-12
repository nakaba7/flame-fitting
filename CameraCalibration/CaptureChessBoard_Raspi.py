import socket
import datetime
import threading
import time
"""
2つのラズパイへのコマンド送信をマルチスレッドで同時に行うスクリプト
"""

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.11.57"  # Raspberry Pi Aのホスト名またはIPアドレスを設定
HOST_B = "192.168.11.56"  # Raspberry Pi Bのホスト名またはIPアドレスを設定
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号

def send_command(command, s):
    s.sendall(command)

def send_commands_simultaneously(command):
    # スレッドを作成して、各Raspberry Piにコマンドを送信
    thread_a = threading.Thread(target=send_command, args=(command, sa))
    thread_b = threading.Thread(target=send_command, args=(command, sb))
    
    thread_a.start()
    thread_b.start()
    
    thread_a.join()
    thread_b.join()

if __name__ == "__main__":
    print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
            sa.connect((HOST_A, PORT))
            sb.connect((HOST_B, PORT))
            count = 0
            while True:
                if count == 60:
                    send_commands_simultaneously(b'q')
                    print("Capture Closed.")
                    break  # ループを抜けて終了

                print(datetime.datetime.now())
                send_commands_simultaneously(b'c')
                print("Signal sent to Raspberry Pi to capture an image.")
                count += 1
                time.sleep(1)
