import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def show_obj(objfilepath):
    # .objファイルの読み込み
    mesh = trimesh.load(objfilepath)
    # メッシュの3次元プロットを作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 頂点座標を取得
    vertices = mesh.vertices
    faces = mesh.faces

    # 頂点座標をfacesに従って再構築
    mesh_collection = Poly3DCollection(vertices[faces], alpha=0.5, facecolor=[0.5, 0.5, 1])

    # メッシュをプロットに追加
    ax.add_collection3d(mesh_collection)

    # 軸ラベルの設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # メッシュのスケールを調整
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # プロットの表示
    plt.show()


if __name__ == '__main__':
    objfilepath = "../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test8_.obj"
    show_obj(objfilepath)