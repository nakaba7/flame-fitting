import cv2
import numpy as np
import os
import argparse
import re
"""
手動で顔の特徴点検出を行うスクリプト. 左目, 右目, 口の順番で行う. 
Usage:
    python FaceAnnotation/manual_annotation.py -p PARTICIPANTNAME -r [False|True]
Args:   
    -p: 参加者名を指定
    -r: インデックスをリセットするかどうかを指定. デフォルトはFalse.
"""

def manual_annotation(img, output_img_path, output_npy_path, index, total, instructions):
    facial_landmarks = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            facial_landmarks.append((x, y))
            cv2.imshow('image', img)

    img_original = img.copy()

    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        img_display = img.copy()
        cv2.putText(img_display, f'Index: {index+1}/{total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_display, instructions, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', img_display)
        cv2.setMouseCallback('image', click_event)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            img = img_original.copy()
            facial_landmarks = []
            print("Resetting landmarks...")
        elif key == ord('b'):
            cv2.destroyAllWindows()
            return 'back'
        elif key == ord('s'):
            cv2.destroyAllWindows()
            return 'stop'
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return 'quit'
        else:
            break
    cv2.destroyAllWindows()

    print("Selected Facial Landmarks:")
    print(facial_landmarks)
    np.save(output_npy_path, np.array(facial_landmarks))
    cv2.imwrite(output_img_path, img)
    print(f'Annotated image saved as {output_img_path}')
    return 'next'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def annotation_onefolder(participant_name, facepart, reset=False):
    input_dir_path = f'FaceImages/{participant_name}/{facepart}'
    facepart_image_list = os.listdir(input_dir_path)
    facepart_image_list.sort(key=natural_sort_key)
    output_img_dir = f'AnnotatedData/{participant_name}_Annotated/Images/{facepart}'
    output_npy_dir = f'AnnotatedData/{participant_name}_Annotated/NPYs/{facepart}'
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    last_index_file = os.path.join(output_img_dir, 'last_index.txt')
    img_count = 0
    i = 0
    if not reset and os.path.exists(last_index_file):
        with open(last_index_file, 'r') as f:
            i = int(f.read().strip())
    previous_i = i
    instructions = 'Commands: r - Reset, b - Back, s - Save and Stop, q - Quit'
    while i < len(facepart_image_list):
        img = facepart_image_list[i]
        input_img_path = os.path.join(input_dir_path, img)
        print(input_img_path)
        input_img = cv2.imread(input_img_path)
        result = manual_annotation(input_img, os.path.join(output_img_dir, f'{img[:-4]}_annotated.jpg'), os.path.join(output_npy_dir, f'{img[:-4]}_annotated.npy'), i, len(facepart_image_list), instructions)
        if result == 'back':
            i = max(0, i - 1)
            if img_count > 0:
                img_count -= 1
        elif result == 'stop':
            with open(last_index_file, 'w') as f:
                f.write(str(i))
            print(f"Annotation stopped and index saved to {last_index_file}.")
            return 'stop'
        elif result == 'quit':
            reset_indices(participant_name)
            print("Annotation process forcibly terminated and indices reset.")
            return 'quit'
        else:
            img_count += 1
            i += 1
    if i != previous_i:
        print(f'{facepart} annotation finished')
    else:
        print(f'No images found in {facepart} folder')
    return 'next'

def reset_indices(participant_name):
    parts = ["lefteye", "righteye", "mouth"]
    for part in parts:
        output_img_dir = f'AnnotatedData/{participant_name}_Annotated/Images/{part}'
        if not os.path.exists(output_img_dir):
            print(f'No index file found for {part}')
            continue
        last_index_file = os.path.join(output_img_dir, 'last_index.txt')
        with open(last_index_file, 'w') as f:
            f.write('0')

def main(args):
    participant_name = args.p
    reset = args.r
    if reset :
        reset_indices(participant_name)
        print("Indices reset.")
        
    for part in ["lefteye", "righteye", "mouth"]:
        if annotation_onefolder(participant_name, part) == 'quit':
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, help='participant name')
    parser.add_argument('-r', default=False, type=bool, help='reset indices')
    main(parser.parse_args())
