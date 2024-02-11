import socket
"""
Raspberry Piにコマンドを送信するスクリプト
"""

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = "nakaba7-a.local"  # Raspberry PiのIPアドレス
port = 46361

server_socket.bind((host, port))

server_socket.listen(1)
print('Waiting for connection...')

client_socket, addr = server_socket.accept()
print('Connected by', addr)

while True:
    cmd = input()  # ユーザーの入力を待つ
    if cmd.lower() == 'q':
        client_socket.sendall(b'q')
        break  # 'q'が入力されたら終了
    elif cmd.lower() == 'c':
        client_socket.sendall(b'Capture')
        print("Signal sent to Raspberry Pi to capture an image.")

client_socket.close()
server_socket.close()