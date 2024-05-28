import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Coordinate_convertor import image2camera_coordinates, camera2image_coordinates, camera2world_coordinates, world2camera_coordinates, mm2pixel, pixel2mm

T = np.load("Parameters/T_mouth_left_eye_left.npy")
distance = np.linalg.norm(T)
mmdist = pixel2mm(distance, 96)
print(mmdist)