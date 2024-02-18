import numpy as np
import glob
from skimage import io

sample_image_path = "img_align_celeba/000001.jpg"
sample_image = io.imread(sample_image_path)
height, width, _ = sample_image.shape
npy_list = glob.glob("output_landmark/*.npy")
print("height: ",height)
print("width: ",width)
print("npy_list: ",npy_list)
for npy in npy_list:
    points = np.load(npy)
    if height > width:
        points /= height
    else:   
        points /= width
    new_npy = npy[:-4] + "_normalized.npy"
    np.save(new_npy, points)
    print(f"Saved {npy}")
    #print("points: \n",points)
    print("points.shape: \n",points.shape)