import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def show_objs_with_aligned_vertices(objfilepath1, objfilepath2):
    # .objファイルの読み込み
    mesh1 = trimesh.load(objfilepath1)
    mesh2 = trimesh.load(objfilepath2)

    # メッシュの頂点座標を取得
    vertices1 = mesh1.vertices
    vertices2 = mesh2.vertices

    # 最初の頂点を取得
    first_vertex1 = vertices1[3573]
    first_vertex2 = vertices2[3573]

    # メッシュ2の頂点を移動させるためのオフセット計算
    offset = first_vertex1 - first_vertex2
    vertices2 += offset

    # メッシュの3次元プロットを作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # メッシュ1の描画
    faces1 = mesh1.faces
    mesh_collection1 = Poly3DCollection(vertices1[faces1], alpha=0.5, facecolor=[0.5, 0.5, 1])
    ax.add_collection3d(mesh_collection1)

    # メッシュ2の描画
    faces2 = mesh2.faces
    mesh_collection2 = Poly3DCollection(vertices2[faces2], alpha=0.5, facecolor=[1, 0.5, 0.5])
    ax.add_collection3d(mesh_collection2)

    # 軸ラベルの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # メッシュのスケールを調整
    scale = np.concatenate([vertices1.flatten(), vertices2.flatten()])
    ax.auto_scale_xyz(scale, scale, scale)

    # プロットの表示
    plt.show()

if __name__ == '__main__':
    sample_list = [346, 361, 336, 371, 341, 245]
    for i in sample_list:
        file_num = f'{i:03d}'
        objfilepath1 = f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/EstimatedModel/00{file_num}_from_params.obj'
        objfilepath2 = f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test00{file_num}.obj'
        show_objs_with_aligned_vertices(objfilepath1, objfilepath2)



