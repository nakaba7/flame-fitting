from lmk_plot import plot_3d
import numpy as np
a=np.load("output_landmark/estimated_3d/quest_3d.npy")
print(a)
print(a.shape)
plot_3d(a)