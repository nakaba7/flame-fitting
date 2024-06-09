import numpy as np
import trimesh
import os

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

if __name__ == '__main__':
    rmse_list = []
    for i in range(400):
        file_num = f'{i:03d}'
        vertices1 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/EstimatedModel/00{file_num}_from_params.obj')
        vertices2 = load_obj(f'../Collect FLAME Landmark/Assets/Objects/FLAMEmodel/test00{file_num}.obj')
        if vertices1 is None or vertices2 is None:
            print(f"Failed to load OBJ file {file_num}")
            continue

        rmse = compute_rmse(vertices1, vertices2)
        rmse_list.append(rmse)
        #print(f"RMSE between the two models: {rmse}")
    print(f"Average RMSE: {np.mean(rmse_list)}")
    print(f"Max RMSE: {np.max(rmse_list)}")
    print(f"Min RMSE: {np.min(rmse_list)}")