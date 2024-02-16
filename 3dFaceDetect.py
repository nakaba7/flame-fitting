import numpy as np
import face_alignment
from skimage import io
from mpl_toolkits import mplot3d 
#%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings 
#import matplotlib
#matplotlib.use("Agg")

"""
顔写真から眉, 目, 鼻, 口の3D特徴点を検出し, npyファイルに保存するスクリプト
"""

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
image_path = 'surprised.jpg'
input = io.imread(image_path)
preds = fa.get_landmarks(input)
#preds=np.array(preds)
#preds=preds[0]
print(preds)
preds = preds[0][17:]
print(np.array(preds).shape)
np.save(f"{image_path[:-4]}.npy",np.array(preds))
#plt.imshow(input)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(preds[:,0], preds[:,1], preds[:,2])
plt.show()
#fig.savefig(image_path[:-4] + "_detected.jpeg")