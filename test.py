import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir
import glob
import time
import argparse



if __name__ == '__main__':
    model_path = './models/generic_model.pkl'
    model = load_model(model_path)           
    print("loaded model from:", model_path)
    outmesh_path = "../Collect FLAME Landmark/Assets/Objects/testmodel1.obj"
    
    while True:
        model.pose[:] = np.random.randn(model.pose.size) * 0.1
        model.pose[0:3] = 0 #回転は0で固定
        model.betas[300:350] = np.random.randn(50)*0.5
        write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
        print('output mesh saved to: ', outmesh_path)
        exit()

        


