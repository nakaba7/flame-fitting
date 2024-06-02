import os
import re

# ディレクトリのパスを指定
#directory_path = 'FaceImages/Nakabayashi/righteye'
directory_path = '../Collect FLAME Landmark/Assets/Objects/FLAMEmodel'
# ディレクトリ内のファイルを一覧取得
for filename in os.listdir(directory_path):
    if filename.startswith('test') and (filename.endswith('_0.jpg') or filename.endswith('_1.jpg') or filename.endswith('_annotated.jpg') or filename.endswith('_annotated.npy')):
        # 現在の番号部分を抽出
        parts = filename.split('_')
        #print(parts)
        number = parts[0][4:]  # 'test'の後の部分を取り出す
        new_number = number.zfill(5)  # 5桁にゼロ埋め
        parts[0] = f'test{new_number}'
        new_filename = '_'.join(parts)
        # ファイルの名前を変更
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        os.rename(old_file, new_file)

        print(f'Renamed: {filename} -> {new_filename}')
    elif filename.startswith('test') and filename.endswith('.obj'):
        parts = filename.split('.')
        number = parts[0][4:]  # 'test'の後の部分を取り出す
        new_number = number.zfill(5)  # 5桁にゼロ埋め

        # 新しいファイル名を作成
        new_filename = f'test{new_number}.obj'

        # ファイルの名前を変更
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        os.rename(old_file, new_file)

        print(f'Renamed: {filename} -> {new_filename}')
    elif filename.startswith('test'):
        parts = filename.split('.')
        number = parts[0][4:]  # 'test'の後の部分を取り出す
        new_number = number.zfill(5)  # 5桁にゼロ埋め

        # 新しいファイル名を作成
        new_filename = f'test{new_number}.jpg'

        # ファイルの名前を変更
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        os.rename(old_file, new_file)

        print(f'Renamed: {filename} -> {new_filename}')

