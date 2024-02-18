import numpy as np
import os

path = sorted(os.listdir("output_landmark/2d"))
outpath = sorted(os.listdir("output_landmark/3d"))
print(path==outpath)
#print("path = ",path)
"""
path = 
a = np.load("output_landmark/2d/000001.npy")
print("before: ",a)
print(a.shape)
a = a[:51]
print("after: ",a)
print(a.shape)
"""