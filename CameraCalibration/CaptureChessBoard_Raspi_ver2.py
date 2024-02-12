import socket
import datetime
import threading
import time

# Raspberry Piのホスト名またはIPアドレスとポート番号
HOST_A = "192.168.11.57"  # Raspberry Pi Aのホスト名またはIPアドレスを設定
HOST_B = "192.168.11.56"  # Raspberry Pi Bのホスト名またはIPアドレスを設定
PORT = 46361              # Raspberry Pi側のスクリプトで使用しているポート番号

# 各Raspberry Piからの応答を受け取る
def receive_response(sock):
    while True:
        data = sock.recv(1024)
        if data == b'q':
            return

# コマンドを送信し、応答を待つ
def send_command_and_wait_for_response(command, sock):
    sock.sendall(command)
    receive_response(sock)

def send_commands_and_wait_for_responses(command, socks):
    threads = []
    for sock in socks:
        thread = threading.Thread(target=send_command_and_wait_for_response, args=(command, sock))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    print("Connected to Raspberry Pi. Press Enter to capture an image, 'q' to quit.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sa, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sb:
        sa.connect((HOST_A, PORT))
        sb.connect((HOST_B, PORT))
        count = 0
        while True:
            if count == 60:
                send_commands_and_wait_for_responses(b'q', [sa, sb])
                print("Capture Closed. Received 'q' from both Raspberry Pis.")
                break  # ループを抜けて終了

            print(datetime.datetime.now())
            send_commands_simultaneously(b'c')
            print("Signal sent to Raspberry Pi to capture an image.")
            count += 1
            time.sleep(1)
