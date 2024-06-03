import os
import shutil
from tqdm import tqdm

def move_obj_files(source_dir, destination_dir):
    # ディレクトリが存在しない場合、作成する
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # source_dir 内の全ファイルをチェックし、.obj ファイルのみをリストアップ
    obj_files = [file_name for file_name in os.listdir(source_dir) if file_name.endswith('.obj')]

    # tqdm を使って進行状況バーを表示しながらファイルを移動
    for file_name in tqdm(obj_files, desc="Moving .obj files"):
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        shutil.move(source_file, destination_file)
# ディレクトリのパスを指定
source_dir = 'output_obj'  # 例: 'C:/path/to/source_dir'
destination_dir = 'obj_for_train'  # 例: 'C:/path/to/destination_dir'

# 関数を呼び出してファイルを移動
move_obj_files(source_dir, destination_dir)
