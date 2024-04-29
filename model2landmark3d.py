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

config = get_config()
flamelayer = FLAME(config)
BATCHSIZE = 8

#poseパラメータの顎に関係する部分を乱数×スケーリングファクターで指定する関数
def create_jaw_array(jaw_opening_factor, jaw_shift_factor, jaw_vertical_factor):
    return np.array([0, 0, 0, np.abs(np.random.randn(1)*jaw_opening_factor), np.random.randn(1) * jaw_shift_factor, np.random.randn(1) * jaw_vertical_factor], dtype=np.float32)

#顔の形状パラメータを0で固定
shape_params = torch.zeros(BATCHSIZE, 100).cuda()

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotation
# pose_params_numpy[:, 3:] : jaw rotation
#[0,0,0,口の開き具合(正が開く),左右方向への曲げ(正が左，負が右), 上下方向への曲げ(正が左，負が右)]

#姿勢，顎のパラメータを指定
pose_params_numpy = np.array([create_jaw_array(0.5, 0.1, 0.1) for _ in range(BATCHSIZE)], dtype=np.float32)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

#pose_params = torch.zeros(BATCHSIZE, 6, dtype=torch.float32).cuda()

#表情パラメータを乱数で指定
expression_params_numpy = np.array([np.random.randn(50) for _ in range(BATCHSIZE)], dtype=np.float32)
expression_params = torch.tensor(expression_params_numpy, dtype=torch.float32).cuda()

#expression_params = torch.zeros(8, 50, dtype=torch.float32).cuda()

flamelayer.cuda()
"""
print("Shape parameters shape: ", shape_params.shape)
print("Pose parameters shape: ", pose_params.shape)
print("Expression parameters shape: ", expression_params.shape)
"""
# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
vertice, landmark = flamelayer(
    shape_params, expression_params, pose_params
)  # For RingNet project

if config.optimize_eyeballpose and config.optimize_neckpose:
    print("Optimizing for eyeball and neck pose")
    neck_pose = torch.zeros(BATCHSIZE, 3).cuda()
    eye_params_numpy = np.array([np.random.randn(6) for _ in range(BATCHSIZE)], dtype=np.float32)
    eye_pose = torch.tensor(eye_params_numpy, dtype=torch.float32).cuda()
    vertice, landmark = flamelayer(
        shape_params, expression_params, pose_params, neck_pose, eye_pose
    )
#輪郭のランドマークを削除
landmark = landmark[:,17:,:]
output_obj_dir = './output'
output_lmk_dir = './output_landmark'

#OBJファイルとして保存
for i in range(BATCHSIZE):
    vertices = vertice[i].detach().cpu().numpy().squeeze()  # verticeはFLAMEモデルからの出力
    faces = flamelayer.faces  # FLAMEモデルの面情報

    # trimeshオブジェクトの作成
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    outmesh_path = join( output_obj_dir, f'output_mesh_{i}.obj')
    # OBJファイルとして保存
    mesh.export(outmesh_path)
    print(landmark[i].shape)
    output_lmk_path = os.path.join(output_lmk_dir,f"lmk_{i}.npy")
    np.save(output_lmk_path, landmark[i].detach().cpu().numpy())
    print('output mesh saved to: ', outmesh_path)

