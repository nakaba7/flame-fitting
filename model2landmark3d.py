import numpy as np
import torch
import trimesh
from os.path import join
from flame_pytorch import FLAME, get_config
from tqdm import tqdm
import os

"""
FLAMEモデルのパラメータを指定して、3Dランドマークを取得するコード
データは loop_num x batchsize 個得られる
生成途中でスケーリングファクターを変更することで, jawの偏りをなくす
outputフォルダにobjファイル, output_landmarkフォルダに3次元np配列のランドマークを保存

Usage:
    python model2landmark3d.py -b [batchsize] -n [loop_num]
Args:
    -b: バッチサイズ
    -n: ループ回数
"""

#poseパラメータの顎に関係する部分を乱数×スケーリングファクターで指定する関数
def create_jaw_array(jaw_opening_factor, jaw_shift_factor, jaw_vertical_factor):
    return np.array([0, 0, 0, np.abs(np.random.randn() * jaw_opening_factor), np.random.randn() * jaw_shift_factor, np.random.randn() * jaw_vertical_factor], dtype=np.float32)

def set_params(batchsize, jaw_opening_factor=0.1, jaw_shift_factor=0.01, jaw_vertical_factor=0.01):
    shape_params = torch.zeros(batchsize, 100).cuda()
    pose_params_numpy = np.zeros((batchsize, 6), dtype=np.float32)
    for i in range(batchsize):
        if i >= batchsize * 0.75:  # バッチの後半1/4でスケーリングファクターを変更
            new_opening_factor = 0.5  # 新しいスケーリングファクター
            new_shift_factor = 0.05
            new_vertical_factor = 0.05
            pose_params_numpy[i] = create_jaw_array(new_opening_factor, new_shift_factor, new_vertical_factor)
        elif i >= batchsize * 0.5:  # バッチの後半1/2でスケーリングファクターを変更
            new_opening_factor = 0.2  # 新しいスケーリングファクター
            new_shift_factor = 0.05
            new_vertical_factor = 0.05
            pose_params_numpy[i] = create_jaw_array(new_opening_factor, new_shift_factor, new_vertical_factor)
        else:
            pose_params_numpy[i] = create_jaw_array(jaw_opening_factor, jaw_shift_factor, jaw_vertical_factor)

    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
    expression_params_numpy = np.array([np.random.randn(50) for _ in range(batchsize)], dtype=np.float32)
    expression_params = torch.tensor(expression_params_numpy, dtype=torch.float32).cuda()
    return shape_params, expression_params, pose_params

def main():
    config = get_config()
    flamelayer = FLAME(config)
    batchsize = config.b
    base_num = 0
    loop_num = config.n
    #output_obj_dir = './output'
    #output_lmk_dir = './output_landmark/3d'
    output_obj_dir = './output/test'
    output_lmk_dir = './output_landmark/3d/test'
    if not os.path.exists(output_obj_dir):
        os.makedirs(output_obj_dir)
        print(f"{output_obj_dir} created")
    if not os.path.exists(output_lmk_dir):
        os.makedirs(output_lmk_dir)
        print(f"{output_lmk_dir} created")

    for i in range(loop_num):
        print(f'loop {i+1}/{loop_num}')
        shape_params, expression_params, pose_params = set_params(batchsize)
        flamelayer.cuda()

        # FLAMEのforward呼び出し
        vertice, landmark = flamelayer(
            shape_params, expression_params, pose_params
        )

        if config.optimize_eyeballpose and config.optimize_neckpose:
            #print("Optimizing for eyeball and neck pose")
            neck_pose = torch.zeros(batchsize, 3).cuda()
            eye_pose = torch.zeros(batchsize, 6).cuda()
            vertice, landmark = flamelayer(
                shape_params, expression_params, pose_params, neck_pose, eye_pose
            )
        #輪郭のランドマークを削除
        landmark = landmark[:,17:,:]
        
        #OBJファイルとして保存
        for j in tqdm(range(batchsize)):
            vertices = vertice[j].detach().cpu().numpy().squeeze()  # verticeはFLAMEモデルからの出力
            faces = flamelayer.faces  # FLAMEモデルの面情報

            # trimeshオブジェクトの作成
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            outmesh_path = join( output_obj_dir, f'output_mesh_{j+base_num}_test.obj')
            #print(outmesh_path)
            # OBJファイルとして保存
            mesh.export(outmesh_path)
            output_lmk_path = join(output_lmk_dir,f"lmk_{j+base_num}_test.npy")
            np.save(output_lmk_path, landmark[j].detach().cpu().numpy())
            #print('output mesh saved to: ', outmesh_path)
        base_num += batchsize
        

if __name__ == '__main__':
    main()
