import cv2
import glob
import os
import argparse
from tqdm import tqdm

def rotate_images(folder_path, angle):
    # フォルダ内の全画像ファイルのパスを取得
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))  # JPGファイルを対象

    # 各画像を指定された角度で回転して保存
    for image_file in tqdm(image_files):
        img = cv2.imread(image_file)

        if angle == 90:
            # 画像を90度回転（反時計回り）
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            # 画像を-90度回転（時計回り）
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            # 画像を180度回転
            rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            print("Unsupported angle. Only 90, -90, and 180 are supported.")
            return

        # 上書き保存
        cv2.imwrite(image_file, rotated_img)

    print(f"All images have been rotated by {angle} degrees.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rotate images in a folder by a specified angle.")
    parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('-a', '--angle', type=int, required=True, choices=[90, -90, 180], help='Rotation angle (90, -90, or 180 degrees).')
    
    args = parser.parse_args()

    folder_path = args.folder
    angle = args.angle

    print("Image folder: ", folder_path)
    print("Rotation angle: ", angle)

    # 画像を回転
    rotate_images(folder_path, angle)
