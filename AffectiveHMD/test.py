import serial
import time

# シリアルポートの設定（適切なポートを指定してください）
SERIAL_PORT = 'COM11'  # Windowsの場合
# SERIAL_PORT = '/dev/ttyACM0'  # LinuxまたはMacの場合
BAUD_RATE = 115200

def main():
    try:
        # シリアルポートを開く
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # シリアル通信の安定のために少し待機
        
        while True:
            # 文字 'c' を送信
            ser.write(b'c')
            print("Sent: 'c'")
            
            

            # 受信データがあるか確認
            if ser.in_waiting > 0:
                # シリアルポートからデータを読み取る
                line = ser.readline().decode('utf-8').rstrip()
                print(f"Received: {line}")

            

    except KeyboardInterrupt:
        print("終了します...")

    finally:
        ser.close()  # シリアルポートを閉じる
        print("シリアルポートを閉じました。")

if __name__ == "__main__":
    main()
