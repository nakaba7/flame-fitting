from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.file import (
    File, 
    CreateDisposition, 
    FileAttributes, 
    DirectoryAccessMask,
    Open
)
import uuid
import os

"""
sambaで共有しているラズパイの写真フォルダの内容をWindows PCの共有フォルダにコピーする.
ラズパイ上で実行．
"""

# Samba共有設定
server = "WINDOWS_PCのIPアドレスまたはホスト名"
username = "ユーザー名"
password = "パスワード"
shared_folder = "共有フォルダ名"
raspberry_photos_path = "/path/to/raspberry/photos"
windows_shared_path = r"\\WINDOWS_PC\shared\photos"


# 接続
connection = Connection(uuid.uuid4(), server, 445)
connection.connect()
session = Session(connection, username, password)
session.connect()

tree = TreeConnect(session, r"\\{}\{}".format(server, shared_folder))
tree.connect()

# フォルダの存在を確認し、存在しなければ作成
directory = Open(tree, windows_shared_path)
try:
    directory.create(
        create_disposition=CreateDisposition.FILE_OPEN_IF,  # 存在する場合は開き、存在しない場合は作成
        desired_access=DirectoryAccessMask.FILE_LIST_DIRECTORY | DirectoryAccessMask.FILE_ADD_SUBDIRECTORY,
        file_attributes=FileAttributes.FILE_ATTRIBUTE_DIRECTORY
    )
finally:
    directory.close()

# フォルダが確認（または作成）された後、ファイルコピーの処理を実行
# ラズベリーパイ上の写真フォルダの内容をリストアップ
for photo_name in os.listdir(raspberry_photos_path):
    photo_path = os.path.join(raspberry_photos_path, photo_name)
    # ファイルを読み込む
    with open(photo_path, 'rb') as photo_file:
        photo_data = photo_file.read()
        # Windows PC上の共有フォルダにファイルをコピー
        with open(os.path.join(windows_shared_path, photo_name), 'wb') as target_file:
            target_file.write(photo_data)

# クリーンアップ
tree.disconnect()
session.logoff()
connection.disconnect()