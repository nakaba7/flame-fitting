#python送信側

import socket
import random
import time

HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
    tvec = [random.random(), random.random(), random.random()]
    result = str(tvec[0])+","+str(tvec[1])+","+str(tvec[2])
    print(result)
    print(type(result))
    client.sendto(result.encode('utf-8'),(HOST,PORT))
    time.sleep(1.0)