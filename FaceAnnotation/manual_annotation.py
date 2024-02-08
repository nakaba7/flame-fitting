import cv2
import numpy as np
import os

"""
Video2jpg.pyで切り出した画像に対して、手動で顔の特徴点をアノテーションするサンプルコード。
アノテーションする枚数はmanual_img_numで指定する。
"""

def manual_annotation(input_path, output_img_path, output_csv_path):
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

    # If you want to save the facial landmarks to a file:
    np.savetxt(output_csv_path, np.array(facial_landmarks), delimiter=',', fmt='%d')

    # Save the image with landmarks
    annotated_image_path = output_img_path
    cv2.imwrite(annotated_image_path, img)
    print(f'Annotated image saved as {annotated_image_path}')
    # The code will open the image in a new window and you can click on the facial landmarks you want to annotate.
    # Clicking on the image will create a red dot and the coordinates will be saved in the 'facial_landmarks' list.
    # Press any key to close the image window and print the selected landmarks coordinates.
    # Uncomment the last two lines to save the coordinates to a text file.

if __name__ == '__main__':
    manual_img_num = 10
    folder_name = 'Nakabayashi_raspicamera'

    input_frames_path = os.listdir(f'FrameImages/{folder_name}')
    output_img_dir = f'FrameImages/{folder_name}_Annotated/Images'
    output_csv_dir = f'FrameImages/{folder_name}_Annotated/CSVs'
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    img_count = 0
    for img in input_frames_path:
        if img_count == manual_img_num:
            print('Manual annotation finished')
            break
        input_img = os.path.join(f'FrameImages/{folder_name}', img)
        manual_annotation(input_img, os.path.join(output_img_dir, f'{img[:-4]}_annotated.jpg'), os.path.join(output_csv_dir, f'{img[:-4]}_annotated.csv'))
        img_count += 1


