import socket
from SerialConnection16 import SerialConnection
import time

host = "192.168.100.42" #Processingで立ち上げたサーバのIPアドレス
port = 10001       #Processingで設定したポート番号
serialConnection = SerialConnection('COM11', 115200)
if __name__ == '__main__':
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成
    socket_client.connect((host, port))                               #サーバに接続

    #socket_client.send('送信するメッセージ'.encode('utf-8')) #データを送信 Python3
    while True:
        serialConnection.UpdateSensorData()
        data = serialConnection.getSensorData()
        comma_separated = ",".join([str(i) for i in data])+ "\n"
        #print(comma_separated)
        socket_client.send(comma_separated.encode('utf-8')) #データを送信
        #time.sleep(0.05)

