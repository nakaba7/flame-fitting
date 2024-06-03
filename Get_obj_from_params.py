import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir
import glob
import argparse

def get_obj_from_params(ex):
    # Load FLAME model (here we load the generic model)
    # Make sure path is correct
    model_path = './models/generic_model.pkl'
    model = load_model(model_path)           # the loaded model object is a 'chumpy' object, check



if __name__ == '__main__':
    model_path = './models/generic_model.pkl'
    model = load_model(model_path)           
    print("loaded model from:", model_path)
    outmesh_dir = './output_obj/Nakabayashi'
    safe_mkdir( outmesh_dir )
    pose_param_list = glob.glob("output_params/Nakabayashi/pose/*.npy")
    expr_param_list = glob.glob("output_params/Nakabayashi/expr/*.npy")
    loop_num = min(len(pose_param_list), len(expr_param_list))
    for i in range(loop_num):
        model.pose[:] = np.load(pose_param_list[i])
        model.betas[300:350] = np.load(expr_param_list[i])
        basename = pose_param_list[i].split("/")[-1].split(".")[0][:-5]
        outmesh_path = join( outmesh_dir, f'{basename}_from_params.obj')
        write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )
        print('output mesh saved to: ', outmesh_path)
