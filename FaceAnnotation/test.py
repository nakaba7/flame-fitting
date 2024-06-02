import cv2
import numpy as np

# 画像と特徴点のファイルパス
image_path = "FaceImages/Nakabayashi/lefteye/test182_0.jpg"
landmarks_path = "AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test182_0_annotated.npy"

# 画像をロード
image = cv2.imread(image_path)
if image is None:
    print(f"Error loading image: {image_path}")
else:
    print(f"Loaded image: {image_path}")

# 特徴点をロード
landmarks = np.load(landmarks_path)
if landmarks.size == 0:
    print(f"Error loading landmarks: {landmarks_path}")
else:
    print(f"Loaded landmarks: {landmarks_path}")
    print(landmarks)

# 特徴点を画像に重ねて表示
for idx, (x, y) in enumerate(landmarks):
    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.putText(image, str(idx + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# 画像を表示
cv2.imshow('Image with Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
