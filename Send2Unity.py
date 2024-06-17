import socket
import time

def send_command():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 65432))
            i=0
            while True:
                command = str(i)
                s.sendall(command.encode('utf-8'))
                print(f"Sent: {command}")
                if command == "q":
                    s.close()
                    break
                data = s.recv(1024)
                print(f"Received ack: {data.decode('utf-8')}")
                if i==0:
                    i=1
                elif i==1:
                    i=0
                

if __name__ == '__main__':
    send_command()
