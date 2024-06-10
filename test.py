import numpy as np
import trimesh
import os
from tqdm import tqdm
def load_obj(file_path):
    if not os.path.exists(file_path):
        return None
    mesh = trimesh.load(file_path)
    return mesh.vertices

def compute_rmse(vertices1, vertices2):
    if vertices1.shape != vertices2.shape:
        raise ValueError("The two models do not have the same number of vertices.")
    
    # Calculate the Euclidean distance between corresponding vertices
    distances = np.linalg.norm(vertices1 - vertices2, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def compute_scale(vertices):
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    scale = max_coords - min_coords
    return min_coords, max_coords, scale

def compute_nrmse(rmse, scale):
    return rmse / np.mean(scale)

if __name__ == '__main__':
    rmse_list = []
    index_list =[]
    for i in tqdm(range(400)):
        file_num = f'{i:03d}'
        vertices1 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/EstimatedModel/00{file_num}_from_params.obj')
        vertices2 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test00{file_num}.obj')
        if vertices1 is None or vertices2 is None:
            print(f"Failed to load OBJ file {file_num}")
            continue
        if vertices1 is not None and vertices2 is not None:
            min_rmse = 100
            min_j = 0
            for j in range(len(vertices1)):
                if vertices1 is not None and vertices2 is not None:
                    # 最初の頂点を取得
                    first_vertex1 = vertices1[j]
                    first_vertex2 = vertices2[j]

                    # メッシュ2の頂点を移動させるためのオフセット計算
                    offset = first_vertex1 - first_vertex2
                    vertices2 += offset
                    rmse = compute_rmse(vertices1, vertices2)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        min_j = j
            rmse_list.append(min_rmse)
            index_list.append(min_j)
            #print("=====================================")
        else:
            print("Failed to load vertices.")
        #rmse = compute_rmse(vertices1, vertices2)
        
        #print(f"RMSE between the two models: {rmse}")
    print(f"Index: {index_list}")
    print(f"Average RMSE: {np.mean(rmse_list)}")
    print(f"Standard Deviation RMSE: {np.std(rmse_list)}")
    print(f"Max RMSE: {np.max(rmse_list)}")
    print(f"Min RMSE: {np.min(rmse_list)}")
    
    """
    sample_list = [346, 361, 336, 371, 341, 245]
    for i in sample_list:
        file_num = f'{i:03d}'
        vertices1 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/EstimatedModel/00{file_num}_from_params.obj')
        vertices2 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test00{file_num}.obj')
        min_rmse = 100
        min_j = 0
        for j in range(len(vertices1)):
            if vertices1 is not None and vertices2 is not None:
                # 最初の頂点を取得
                first_vertex1 = vertices1[j]
                first_vertex2 = vertices2[j]

                # メッシュ2の頂点を移動させるためのオフセット計算
                offset = first_vertex1 - first_vertex2
                vertices2 += offset
                rmse = compute_rmse(vertices1, vertices2)
                if rmse < min_rmse:
                    min_rmse = rmse
                    min_j = j
                #print(min_coords1, max_coords1, scale1)
                #print(min_coords2, max_coords2, scale2)
                #print(f"RMSE {i}: {rmse}")
                
                #print("=====================================")
        print(f"Min RMSE {i}: {min_rmse}")
        print(f"Min J {i}: {min_j}")
        print("=====================================")
    """
    #rmse = compute_rmse(vertices1, vertices2)
    #rmse_list.append(rmse)
    #print(f"RMSE between the two models: {rmse}")
    """
    for i in range(400):
        file_num = f'{i:03d}'
        vertices1 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/EstimatedModel/00{file_num}_from_params.obj')
        vertices2 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test00{file_num}.obj')
        if vertices1 is None or vertices2 is None:
            print(f"Failed to load OBJ file {file_num}")
            continue
        if vertices1 is not None and vertices2 is not None:
            min_rmse = 100
            min_j = 0
            for j in range(len(vertices1)):
                if vertices1 is not None and vertices2 is not None:
                    # 最初の頂点を取得
                    first_vertex1 = vertices1[j]
                    first_vertex2 = vertices2[j]

                    # メッシュ2の頂点を移動させるためのオフセット計算
                    offset = first_vertex1 - first_vertex2
                    vertices2 += offset
                    rmse = compute_rmse(vertices1, vertices2)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        min_j = j
            print("=====================================")
        else:
            print("Failed to load vertices.")
        #rmse = compute_rmse(vertices1, vertices2)
        #rmse_list.append(rmse)
        #print(f"RMSE between the two models: {rmse}")
    print(f"Average RMSE: {np.mean(rmse_list)}")
    print(f"Average NRMSE: {np.mean(nrmse_list)}")
    print(f"Max RMSE: {np.max(rmse_list)}")
    print(f"Min RMSE: {np.min(rmse_list)}")
    """