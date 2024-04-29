"""
FLAMEモデルのパラメータを指定して、3Dランドマークを取得するコード
生成途中でスケーリングファクターを変更することで, jawの偏りをなくす
"""
import numpy as np
import pyrender
import torch
import trimesh

from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir

from flame_pytorch import FLAME, get_config
from datetime import datetime
import os
from tqdm import tqdm

#poseパラメータの顎に関係する部分を乱数×スケーリングファクターで指定する関数
def create_jaw_array(jaw_opening_factor, jaw_shift_factor, jaw_vertical_factor):
    return np.array([0, 0, 0, np.abs(np.random.randn() * jaw_opening_factor), np.random.randn() * jaw_shift_factor, np.random.randn() * jaw_vertical_factor], dtype=np.float32)

#顔形状，表情，姿勢パラメータを指定
def set_params(batchsize, jaw_opening_factor=0.5, jaw_shift_factor=0.1, jaw_vertical_factor=0.1):
    shape_params = torch.zeros(batchsize, 100).cuda()
    
    # pose_params_numpy[:, :3] : global rotation
    # pose_params_numpy[:, 3:] : jaw rotation
    #[0,0,0,口の開き具合(正が開く),左右方向への曲げ(正が左，負が右), 上下方向への曲げ(正が左，負が右)]

    #姿勢，顎のパラメータを指定
    pose_params_numpy = np.array([create_jaw_array(jaw_opening_factor, jaw_shift_factor, jaw_vertical_factor) for _ in range(batchsize)], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

    #pose_params = torch.zeros(batchsize, 6, dtype=torch.float32).cuda()

    #表情パラメータを乱数で指定
    expression_params_numpy = np.array([np.random.randn(50) for _ in range(batchsize)], dtype=np.float32)
    expression_params = torch.tensor(expression_params_numpy, dtype=torch.float32).cuda()

    #expression_params = torch.zeros(8, 50, dtype=torch.float32).cuda()

    return shape_params, expression_params, pose_params

def main():
    config = get_config()
    flamelayer = FLAME(config)
    batchsize = config.b

    shape_params, expression_params, pose_params = set_params(batchsize)
    flamelayer.cuda()

    # FLAMEのforward呼び出し
    vertice, landmark = flamelayer(
        shape_params, expression_params, pose_params
    )

    if config.optimize_eyeballpose and config.optimize_neckpose:
        print("Optimizing for eyeball and neck pose")
        neck_pose = torch.zeros(batchsize, 3).cuda()
        eye_pose = torch.zeros(batchsize, 6).cuda()
        vertice, landmark = flamelayer(
            shape_params, expression_params, pose_params, neck_pose, eye_pose
        )
    #輪郭のランドマークを削除
    landmark = landmark[:,17:,:]
    output_obj_dir = './output'
    output_lmk_dir = './output_landmark/3d'

    #OBJファイルとして保存
    for i in tqdm(range(batchsize)):
        vertices = vertice[i].detach().cpu().numpy().squeeze()  # verticeはFLAMEモデルからの出力
        faces = flamelayer.faces  # FLAMEモデルの面情報

        # trimeshオブジェクトの作成
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        outmesh_path = join( output_obj_dir, f'output_mesh_{i}.obj')
        # OBJファイルとして保存
        mesh.export(outmesh_path)
        output_lmk_path = os.path.join(output_lmk_dir,f"lmk_{i}.npy")
        np.save(output_lmk_path, landmark[i].detach().cpu().numpy())
        #print('output mesh saved to: ', outmesh_path)

if __name__ == '__main__':
    main()
