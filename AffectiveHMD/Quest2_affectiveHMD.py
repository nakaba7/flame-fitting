import serial
import csv
import os
import keyboard
"""
データ収集に使う関数の定義
"""


def setup(output_dir, port='COM11', baudrate=115200):
    """
    シリアルポートとCSVファイルを設定する関数。
    
    Args:
        output_dir (str): CSVファイルを保存するディレクトリのパス。
        port (str): シリアルポート名 (例: 'COM11')。
        baudrate (int): ボーレート (例: 115200)。
    
    Returns:
        ser (serial.Serial): シリアルポートオブジェクト。
        csv_file_path (str): CSVファイルのパス。
    """
    # シリアルポートの設定
    ser = serial.Serial(port, baudrate)
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_file_path = os.path.join(output_dir, 'sensor_data.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        print(f"CSVファイルを作成しました: {csv_file_path}")
    
    return ser, csv_file_path

def receive_data(ser):
    """
    シリアルポートからデータを1回分受信し, コンソールに表示する関数. テスト用.
    """
    try:
        line = ser.readline().decode('utf-8', errors='ignore').rstrip() 
        if line:
            dataList = line.split(',')
            # 数字のみのリストに変換。無効なデータは無視
            dataList = [int(x) for x in dataList if x.isdigit()]
        else:
            print("データが送信されていません。")
            dataList = None
        return dataList
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None
    
def write_csv(ser, csvwriter):
    """
    シリアルポートからデータを受信し、CSVファイルに1行書き込む関数。
    
    Args:
        ser (serial.Serial): シリアルポートオブジェクト。
        csvwriter: CSVファイルに書き込むためのcsv.writerオブジェクト。
    """
    error_list = [0 for i in range(16)]
    try:
        #with open(csv_file_path, 'a', newline='') as csvfile:
        #csvwriter = csv.writer(csv_file, delimiter=',')
        line = ser.readline().decode('utf-8', errors='ignore').rstrip()  # エラーを無視するデコード
        if line:
            dataList = line.split(',')
            # 数字のみのリストに変換。無効なデータは無視
            dataList = [int(x) for x in dataList if x.isdigit()]
            if dataList:  # データリストが空でない場合
                csvwriter.writerow(dataList)
                #print(dataList)
        else:
            print("データが送信されていません。")
            csvwriter.writerow(error_list) #行ずれを回避するため
                    
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    output_dir = '../sensor_values/test'
    ser, csv_file_path = setup(output_dir)
    try:
        while True:
            if keyboard.is_pressed('c'):
                ser.reset_input_buffer()  # シリアルバッファをクリア
                ser.write(b'c')  # Arduinoに合図を送る
                # 確実にデータが返ってくるまで待機
                while ser.in_waiting == 0:
                    pass
                receive_data(ser, csv_file_path)
                
    except KeyboardInterrupt:
        print("プログラムを終了します。")
    finally:
        ser.close()  # シリアルポートを閉じる
