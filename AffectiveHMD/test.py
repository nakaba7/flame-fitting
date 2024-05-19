from ReceiveSensorData import setup, receive_data  
import time
if __name__ == "__main__":
    output_dir = 'sensor_values/test'
    ser, csv_file_path = setup(output_dir)
    
    # 他のファイルからデータを受信してCSVに保存する例
    try:
        while True:
            receive_data(ser, csv_file_path)
            #print("Received data")
            time.sleep(0.11)
    except KeyboardInterrupt:
        print("プログラムを終了します。")
    finally:
        ser.close()  # シリアルポートを閉じる
