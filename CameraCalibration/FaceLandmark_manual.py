import cv2
import numpy as np
import os
import argparse
import sys
import signal

"""
顔画像に対して特徴点を手動でアノテーションするスクリプト

-特徴点の数はLAMDMARK_NUMで指定し, その数だけ特徴点をクリックすることでアノテーションが完了する．
それ以外の場合は再度クリックを求める．
-顔が写っていない画像はqを押すことでスキップできる.
-特徴点は両画像で共通の場所につける必要がある．
-特徴点の座標はpoints_a_{name}.npy, points_b_{name}.npyに保存される.
-特徴点を付けた画像はFaceImages/{name}/{name}_Annotated/に保存される.

Usage:
    python FaceLandmark_manual.py -n [name]
    - name: 参加者の名前
"""
# アノテーションする特徴点の数
LAMDMARK_NUM = 11
ANNOTATION_NUM = 3

def manual_face_annotation(img_a_path, img_b_path, name, output_img_dir):
    #画像1枚分の特徴点を格納するリスト
    #tmp_point_a = []
    #tmp_point_b = []
    # 画像を読み込む
    def click_event_a(event, x, y, flags, params):
        # 左ボタンがクリックされた場合
        if event == cv2.EVENT_LBUTTONDOWN:
            # 座標をリストに追加
            tmp_point_a.append((x, y))
            # クリックされた位置に印をつける
            cv2.circle(img_a, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image A", img_a)
    def click_event_b(event, x, y, flags, params):
        # 左ボタンがクリックされた場合
        if event == cv2.EVENT_LBUTTONDOWN:
            # 座標をリストに追加
            tmp_point_b.append((x, y))
            # クリックされた位置に印をつける
            cv2.circle(img_b, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image B", img_b)
    
    # 画像Aに対する処理
    window_name_a = 'Image A'
    while True:# 特徴点がLANDMARK_NUMと同じ数になるまで繰り返す．qを押すとその画像をスキップ．
        tmp_point_a = []
        img_a = cv2.imread(f'FaceImages/{name}/{name}_a/{img_a_path}')
        cv2.imshow(window_name_a, img_a)
        cv2.setMouseCallback(window_name_a, click_event_a)
        key = cv2.waitKey(0) & 0xFF  # キーボード入力を待ち、ASCIIコードに変換
        if key == ord('q'):
            print("Skip this image")
            return 0, 0
        elif len(tmp_point_a) == LAMDMARK_NUM:
            break
        else:
            print(f"Please click {LAMDMARK_NUM} points")
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()

    # 画像Bに対する処理
    window_name_b = 'Image B'
    while True:# 特徴点がLANDMARK_NUMと同じ数になるまで繰り返す．qを押すとその画像をスキップ．
        tmp_point_b = []
        img_b = cv2.imread(f'FaceImages/{name}/{name}_b/{img_b_path}')
        cv2.imshow(window_name_b, img_b)
        cv2.setMouseCallback(window_name_b, click_event_b)
        key = cv2.waitKey(0) & 0xFF  # キーボード入力を待ち、ASCIIコードに変換
        if key == ord('q'):
            print("Skip this image")
            cv2.destroyAllWindows()
            return 0, 0
        elif len(tmp_point_b) == LAMDMARK_NUM:
                break
        else:
            print(f"Please click {LAMDMARK_NUM} points")
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(output_img_dir, f'{img_a_path[:-4]}_a_annotated.jpg'), img_a)
    cv2.imwrite(os.path.join(output_img_dir, f'{img_b_path[:-4]}_b_annotated.jpg'), img_b)
    print(tmp_point_a, tmp_point_b)
    return tmp_point_a, tmp_point_b

def main(args):
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    name = args.n
    # 特徴点を格納するリスト
    points_a = []
    points_b = []
    input_frames_path_a = os.listdir(f'FaceImages/{name}/{name}_a/')
    input_frames_path_b = os.listdir(f'FaceImages/{name}/{name}_b/')
    output_img_dir = f'FaceImages/{name}/{name}_Annotated/'
    os.makedirs(output_img_dir, exist_ok=True)
    count = 0 #画像の枚数
    for img_path_a, img_path_b in zip(input_frames_path_a, input_frames_path_b):
        if count ==ANNOTATION_NUM:
            break
        print(img_path_a, img_path_b)
        tmp_point_a, tmp_point_b = manual_face_annotation(img_path_a, img_path_b, name, output_img_dir)
        if tmp_point_a == 0 and tmp_point_b == 0:  # キャンセルされた場合
            continue
        count += 1
        points_a.append(tmp_point_a)
        points_b.append(tmp_point_b)
        
    # 座標をnpyファイルに保存
    points_a_path = f'Points/points_a_{name}.npy'
    points_b_path = f'Points/points_b_{name}.npy'
    np.save(points_a_path, np.array(points_a))
    np.save(points_b_path, np.array(points_b))
    print("Shapes: ",np.shape(points_a), np.shape(points_b))
    print("Saved", points_a_path,"and", points_b_path, ".")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, help='Participant name')
    main(parser.parse_args())
