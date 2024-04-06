import cv2
import glob
import os
import sys

def rotate_images(folder_path):
    # フォルダ内の全画像ファイルのパスを取得
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))  # JPGファイルを対象

    # 各画像を指定された角度で回転して保存
    for image_file in image_files:
        img = cv2.imread(image_file)
        #print(image_file)
        if image_file[-5] == "0":
            # 画像を-90度回転（時計回り）
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif image_file[-5] == "1":
            # 画像を90度回転（反時計回り）
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            print("Unsupported angle. Only 90 and -90 are supported.")
            return
        # 上書き保存
        cv2.imwrite(image_file, rotated_img)

    print(f"All images have been rotated.")

if __name__ == '__main__':
    # 画像が保存されているフォルダのパス
    #folder_path = 'ChessBoard_a'
    folder_path = 'ChessBoard_b'
    print("Image folder : ",folder_path)

    # ユーザーからの入力を取得
    angle = int(input("Enter the rotation angle (90 or -90): "))

    # 画像を回転
    rotate_images(folder_path, angle)
