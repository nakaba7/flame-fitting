import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir
import glob
import os

def get_obj_from_params(ex):
    # Load FLAME model (here we load the generic model)
    # Make sure path is correct
    model_path = './models/generic_model.pkl'
    model = load_model(model_path)           # the loaded model object is a 'chumpy' object, check

if __name__ == '__main__':
    model_path = './models/generic_model.pkl'
    model = load_model(model_path)           
    print("loaded model from:", model_path)
    outmesh_dir = "../Collect FLAME Landmark/Assets/Objects/Estimated"
    if not os.path.exists(outmesh_dir):
        os.makedirs(outmesh_dir)
    
    #pose_param_list = glob.glob("output_params/Nakabayashi/pose/*.npy")
    #expr_param_list = glob.glob("output_params/Nakabayashi/expr/*.npy")
    estimated_param_list = glob.glob("output_params/estimated/*.npy")
    for i in range(len(estimated_param_list)):
        param_npy = np.load(estimated_param_list[i])
        param_npy = param_npy.squeeze()
        #print(param_npy.shape)
        #print(f"loaded {estimated_param_list[i]}")
        model.pose[:] = param_npy[50:]
        model.betas[300:350] = param_npy[:50]
        basename = estimated_param_list[i].split("/")[-1].split(".")[0]
        outmesh_path = join( outmesh_dir, f'{basename}_from_params.obj')
        write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
        print('output mesh saved to: ', outmesh_path)
