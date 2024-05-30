import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Coordinate_convertor import image2camera_coordinates, camera2image_coordinates, camera2world_coordinates, world2camera_coordinates, mm2pixel, pixel2mm
from lmk_plot import plot_2d
camera_mouth_image_points = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/mouth/a1_annotated.npy")
plot_2d(camera_mouth_image_points)
