import processing.net.*;

int port = 10001; // 適当なポート番号を設定

Server server;

void setup() {
  size(800, 400);  // ウィンドウのサイズを設定
  server = new Server(this, port);
  println("server address: " + server.ip()); // IPアドレスを出力
}
void drawAxes() {
  stroke(0);
  // Y軸
  line(50, 50, 50, height - 50);
  // X軸
  line(50, height - 50, width - 50, height - 50);

  // Y軸の目盛りとラベル（0から1000まで）
  int yTicks = 10; // 縦軸の目盛りの数
  for (int i = 0; i <= yTicks; i++) {
    float y = map(i, 0, yTicks, height - 50, 50);
    line(45, y, 55, y); // 目盛り
    fill(0);
    text(i * 100, 30, y + 5); // ラベル
  }

  // X軸の目盛りとラベル（センサ番号）
  int numSensors = 16; // センサの数
  float barWidth = (width - 100) / numSensors; // 棒の幅を計算
  for (int i = 0; i < numSensors; i++) {
    float x = 50 + i * barWidth + barWidth / 2; // 棒の中心位置
    line(x, height - 45, x, height - 55); // 目盛り
    fill(0);
    text(i + 1, x - 5, height - 30); // ラベル（センサ番号）
  }
}

// 棒グラフを描画する関数
void drawBars(float[] values) {
  int numBars = 16;
  float barWidth = (width - 100) / numBars; // 棒の幅を計算

  for (int i = 0; i < numBars; i++) {
    float x = 50 + i * barWidth; // 棒のX座標
    float y = map(values[i], 0, 1000, height - 50, 50); // 棒の高さをマッピング
    rect(x, y, barWidth - 5, height - 50 - y); // 棒を描画
  }
}


void draw() {
  background(255);  // 背景を白に設定
  drawAxes(); // 軸を描画
  Client client = server.available();
  if (client != null) {
    String whatClientSaid = "";
    String temp;
    // 使用可能なデータを全て読み込む
    while ((temp = client.readStringUntil('\n')) != null) {
      whatClientSaid = temp.trim(); // 最後のデータを保持
    }
    // 最新のデータに基づいて処理
    if (!whatClientSaid.equals("")) {
      String[] sensors = whatClientSaid.split(",");
      if (sensors.length == 16) {  // データが16次元であることを確認
        float[] sensors_float = new float[16];
        for (int i = 0; i < 16; i++) {
          try {
            sensors_float[i] = Float.parseFloat(sensors[i]);
          } catch (NumberFormatException e) {
            sensors_float[i] = 0;  // 無効なデータを0に置き換える
          }
        }
        drawBars(sensors_float); // 棒グラフを描画
      }
    }
  }
}
