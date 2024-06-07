import numpy as np
import chumpy as ch
from os.path import join, basename
import argparse
import scipy.sparse as sp
import glob
import sys
import time
import os
# 親ディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# 親ディレクトリをsys.pathに追加
sys.path.append(parent_dir)
from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor

"""
3次元ランドマークをFLAMEモデルにフィッティングし、objファイルとして保存するスクリプト

Usage:
    python Convert/lmk3d_2_obj.py -f [3Dランドマークディレクトリ] -o [出力ディレクトリ] (-r)
    python Convert/lmk3d_2_obj.py -i [入力ファイルパス] -o [出力ディレクトリ] (-r)
Args:
    -f: 3次元ランドマークを含むディレクトリ
    -o: 出力ディレクトリ
    -i: 入力ファイル(指定した場合、そのファイルのみを処理する)
    -r: 既存のファイルを上書きするかどうか
"""
time_list = []
def fit_lmk3d( lmk_3d,                   # input landmark 3d
            model,                       # model
            lmk_face_idx, lmk_b_coords,  # landmark embedding
            weights,                     # weights for the objectives
            shape_num=300, expr_num=100, opt_options=None ):
    
    """ function: fit FLAME model to 3D landmarks

    input: 
        lmk_3d: input landmark 3D, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    pose_idx       = np.union1d(np.arange(3), np.arange(6,9)) # global rotation and jaw rotation
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    free_variables = [ model.trans, model.pose[pose_idx], model.betas[used_idx] ] 
    
    # weights
    #print("fit_lmk3d(): use the following weights:")
    #for kk in weights.keys():
    #    print("fit_lmk3d(): weights['%s'] = %f" % ( kk, weights[kk] ))

    # objectives
    # lmk
    lmk_err = landmark_error_3d( mesh_verts=model, 
                                 mesh_faces=model.f, 
                                 lmk_3d=lmk_3d, 
                                 lmk_face_idx=lmk_face_idx, 
                                 lmk_b_coords=lmk_b_coords, 
                                 weight=weights['lmk'] )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation
    objectives = {}
    objectives.update( { 'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err } ) 

    # options
    if opt_options is None:
        print("fit_lmk3d(): no 'opt_options' provided, use default settings.")
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
        opt_options['maxiter'] = 2000
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass
    
    # optimize
    # step 1: rigid alignment
    timer_start = time.time()
    #print("\nstep 1: start rigid fitting...")
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[0:3] ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    #timer_end = time()
    #print("step 1: fitting done, in %f sec\n" % ( timer_end - timer_start ))

    # step 2: non-rigid alignment
    #timer_start = time()
    #print("step 2: start non-rigid fitting...")    
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time.time()
    print("fitting done, in %f sec\n" % ( timer_end - timer_start ))
    time_list.append(timer_end - timer_start)
    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return model.r, model.f, parms

# -----------------------------------------------------------------------------

def run_fitting(lmk_path, model, lmk_face_idx, lmk_b_coords, weights, shape_num, expr_num, opt_options, output_obj_dir, output_param_dir, participant_name, overwrite_flag=False):
    print("load:", lmk_path)
    # write result
    filename = basename(lmk_path)[:-4]
    output_expr_dir = join(output_param_dir, participant_name,'expr')
    output_pose_dir = join(output_param_dir, participant_name,'pose')
    
    if not os.path.exists(output_expr_dir):
        os.makedirs(output_expr_dir)
    if not os.path.exists(output_pose_dir):
        os.makedirs(output_pose_dir)

    output_obj_path = join(output_obj_dir, f'{filename}.obj' )
    output_expr_path = join(output_expr_dir, f'{filename}_expr.npy' )
    output_pose_path = join(output_pose_dir, f'{filename}_pose.npy' )
    if os.path.exists(output_obj_path) and not overwrite_flag:
        print("already exists:", output_obj_path)
        print("------------------------------------------")
        return
    lmk_3d = np.load(lmk_path)
    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d,                                         # input landmark 3d
                                       model=model,                                           # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                       weights=weights,                                       # weights for the objectives
                                       shape_num=shape_num, expr_num=expr_num, opt_options=opt_options ) # options
    #print(parms['betas'][300:300+min(100,expr_num)])
    #print(len(parms['pose'])) #15
    #print(len(parms['betas'])) #400
    expr_param_np = np.array(parms['betas'][300:300+min(100,expr_num)])
    pose_param_np = np.array(parms['pose'])
    #print(expr_param_np)
    #print(pose_param_np)
    #print("write Obj to:", output_obj_path)
    np.save(output_expr_path, expr_param_np)
    np.save(output_pose_path, pose_param_np)
    print("write Expr to:", output_expr_path)
    print("write Pose to:", output_pose_path)
    print("------------------------------------------")
    #write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_obj_path, verbose=False )

def main(args):
    lmk_dir = args.f
    output_obj_dir = args.o
    overwrite_flag = args.r
    output_param_dir = args.p
    participant_name = args.n
    safe_mkdir(output_obj_dir)

    # model
    model_path = './models/generic_model.pkl' # change to 'female_model.pkl' or 'male_model.pkl', if gender is known
    model = load_model(model_path)       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details

    # landmark embedding
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)

    # weights
    weights = {}
    # landmark term
    weights['lmk']   = 1.0   
    # shape regularizer (weight higher to regularize face shape more towards the mean)
    weights['shape'] = 1.0
    # expression regularizer (weight higher to regularize facial expression more towards the mean)
    weights['expr']  = 1e-3 #default 1e-3
    # regularization of head rotation around the neck and jaw opening (weight higher for more regularization)
    weights['pose']  = 1e-2 #default 1e-2
    
    # number of shape and expression parameters (we do not recommend using too many parameters for fitting to sparse keypoints)
    shape_num = 100
    expr_num = 50

    # optimization options
    opt_options = {}
    opt_options['disp']    = 0 #1にするとパラメータが表示される
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 2000
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver
    if args.i:
        run_fitting(args.i, model, lmk_face_idx, lmk_b_coords, weights, shape_num, expr_num, opt_options, output_obj_dir, output_param_dir, participant_name, overwrite_flag)
        return
    all_lmk = glob.glob(f'{lmk_dir}/*.npy')
    lmk_num = len(all_lmk)
    for i, lmk_path in enumerate(all_lmk):
        print(f"{i+1}/{lmk_num}")
        run_fitting(lmk_path, model, lmk_face_idx, lmk_b_coords, weights, shape_num, expr_num, opt_options, output_obj_dir, output_param_dir, participant_name, overwrite_flag)
        #print(time_list)
    print("average time:", sum(time_list)/len(time_list))
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='output_landmark/estimated_3d', type=str,  help='directory of the 3D landmarks.')
    parser.add_argument('-i', type=str,  help='input file path')
    parser.add_argument('-o', default='../Collect FLAME Landmark/Assets/Objects/FLAMEmodel', type=str,  help='output obj directory path')
    parser.add_argument('-p', default='output_params', type=str,  help='output params directory path')
    parser.add_argument('-r', action='store_true', help='overwrite existing files or not.')
    parser.add_argument('-n', type=str, help='name of a participant.')
    main(parser.parse_args())