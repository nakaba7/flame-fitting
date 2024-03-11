import serial
import csv
import time
"""
Arduinoから送信されたセンサーデータを受信し、CSVファイルに書き込む. ローカルで実行する.
"""

# シリアルポートの設定
ser = serial.Serial('COM9', 9600) # COMポート名を適切に設定

try:
    with open('sensor_data.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        while True:
            # データが送信された場合のみ読み取る
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').rstrip()  # エラーを無視するデコード
                # 有効なデータのみを処理
                if line:
                    dataList = line.split(',')
                    # 数字のみのリストに変換。無効なデータは無視
                    dataList = [int(x) for x in dataList if x.isdigit()]
                    if dataList:  # データリストが空でない場合
                        csvwriter.writerow(dataList)
                        print(dataList)
            #time.sleep(0.1)  # 0.1秒待つ
except KeyboardInterrupt:
    print("プログラムを終了します。")
    ser.close()  # シリアルポートを閉じる
