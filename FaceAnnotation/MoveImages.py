import os
import shutil
"""
画像を左目、右目、口のフォルダに移動するスクリプト. 

usage:
    python FaceAnnotation/MoveImages.py
"""
# ディレクトリを空にする
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 対象ディレクトリのパス
source_directory = 'FaceImages/Nakabayashi'
mouth_directory = 'FaceImages/Nakabayashi/mouth'
lefteye_directory = 'FaceImages/Nakabayashi/lefteye'
righteye_directory = 'FaceImages/Nakabayashi/righteye'

for directory in [mouth_directory, lefteye_directory, righteye_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

clear_directory(mouth_directory)
clear_directory(lefteye_directory)
clear_directory(righteye_directory)

# 対象ディレクトリから全てのファイルを取得
files = os.listdir(source_directory)

# .jpgファイルをフィルタリング
jpg_files = [f for f in files if f.endswith('.jpg')]

for file in jpg_files:
    # ファイル名の末尾を取得
    if file[-6:-4] == '_0':
        # lefteyeフォルダに移動
        shutil.move(os.path.join(source_directory, file), os.path.join(lefteye_directory, file))
    elif file[-6:-4] == '_1':
        # righteyeフォルダに移動
        shutil.move(os.path.join(source_directory, file), os.path.join(righteye_directory, file))
    else:
        # mouthフォルダに移動
        shutil.move(os.path.join(source_directory, file), os.path.join(mouth_directory, file))

print("ファイルの移動が完了しました。")
