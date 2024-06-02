import numpy as np
import chumpy as ch
from os.path import join, basename
import argparse
import scipy.sparse as sp
import glob
import sys
import os
from Lmk_plot import plot_2d, plot_3d

aligned_3d_lmk = np.load("output_landmark/aligned_3d/test58.npy")
estimated_2d_lmk = np.load("output_landmark/estimated_2d/test58.npy")
estimated_3d_lmk = np.load("output_landmark/estimated_3d/test58.npy")
plot_3d(aligned_3d_lmk)
plot_2d(estimated_2d_lmk)
plot_3d(estimated_3d_lmk)
