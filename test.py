import socket
import datetime
import time
import argparse
import sys
import os
import csv

csv_file_path = "sensor_values/participant1.csv"
timestamp = datetime.datetime.now().strftime("%m%d%H%M")
# ファイル名にタイムスタンプを追加
csv_file_path = csv_file_path.replace(".csv", f"_{timestamp}.csv")

print(f"Opened csv file: {csv_file_path}")
