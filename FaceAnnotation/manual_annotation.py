import cv2
import numpy as np
import os
import argparse

"""
顔画像に対して、手動で顔の特徴点をアノテーションするサンプルコード。
コマンドライン引数でフォルダ名とアノテーションする画像の枚数を指定する。

usage:
$ python manual_annotation.py -f [folder_name] -n [number_of_images]
-f: フォルダ名
-n: アノテーションする画像の枚数
"""

def manual_annotation(input_path, output_img_path, output_npy_path):
    # Initialize list to store coordinates of facial landmarks
    facial_landmarks = []

    # Function to display the image and capture points
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            facial_landmarks.append((x, y))
            cv2.imshow('image', img)

    # Load the image
    img = cv2.imread(input_path)

    # Create a named window
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    # Set the window to fullscreen
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('image', img)

    # Set mouse callback function
    cv2.setMouseCallback('image', click_event)

    # Display the image until a key is pressed
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

    # Print the selected points
    print("Selected Facial Landmarks:")
    print(facial_landmarks)

    # Save the facial landmarks to a .npy file
    np.save(output_npy_path, np.array(facial_landmarks))

    # Save the image with landmarks
    annotated_image_path = output_img_path
    cv2.imwrite(annotated_image_path, img)
    print(f'Annotated image saved as {annotated_image_path}')

def main(args):
    folder_name = args.f
    manual_img_num = args.n
    input_Imagess_path = os.listdir(f'{folder_name}')
    output_img_dir = f'FaceData/{folder_name}_Annotated/Images'
    output_npy_dir = f'FaceData/{folder_name}_Annotated/NPYs'
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    img_count = 0
    for img in input_Imagess_path:
        if img_count == manual_img_num:
            print('Manual annotation finished')
            break
        input_img = os.path.join(f'{folder_name}', img)
        manual_annotation(input_img, os.path.join(output_img_dir, f'{img[:-4]}_annotated.jpg'), os.path.join(output_npy_dir, f'{img[:-4]}_annotated.npy'))
        img_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='hoge', type=str,  help='folder name of the images to be annotated')
    parser.add_argument('-n', default=10, type=int,  help='number of images to be annotated')
    main(parser.parse_args())
