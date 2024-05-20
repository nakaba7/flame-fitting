import os
import shutil
"""
画像を左目、右目、口のフォルダに移動するスクリプト
"""

# 対象ディレクトリのパス
source_directory = '../FaceImages/Nakabayashi'
mouth_directory = '../FaceImages/Nakabayashi/mouth'
lefteye_directory = '../FaceImages/Nakabayashi/lefteye'
righteye_directory = '../FaceImages/Nakabayashi/righteye'
if not os.path.exists(mouth_directory):
    os.makedirs(mouth_directory)
if not os.path.exists(lefteye_directory):
    os.makedirs(lefteye_directory)
if not os.path.exists(righteye_directory):
    os.makedirs(righteye_directory)

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
