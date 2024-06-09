import numpy as np
import glob

params = glob.glob("output_params/estimated/*.npy")
for i in range(len(params)):
    param = np.load(params[i])
    #print(param.shape)
    param = param.squeeze()
    print(param[50:53])
    print("=============================================")