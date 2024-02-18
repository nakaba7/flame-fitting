from skimage import io
import numpy as np
import glob

images = glob.glob('img_align_celeba/*.jpg')
print(len(images))
for i in range(len(images)):
    print(images[i])
    if i > 3:
        break