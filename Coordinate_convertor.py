import numpy as np

def image2camera_coordinates(image_points, z_value, camera_matrix, is_mouthcamera=False):
    """
    画像座標系からカメラ座標系への変換を行う。z_valueはカメラ座標系のz座標をピクセル単位で指定する。z座標はz_valueで固定される。
    """
    # Add a third coordinate of 1 to the image points (homogeneous coordinates)
    ones = np.ones((image_points.shape[0], 1))
    image_points_homogeneous = np.hstack((image_points, ones))

    # Calculate the inverse of the camera matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Convert image coordinates to camera coordinates
    camera_coords_homogeneous = np.dot(camera_matrix_inv, image_points_homogeneous.T).T
    
    # Scale the camera coordinates by the z value
    camera_coords = camera_coords_homogeneous * z_value
    
    # Add the z value as the third coordinate
    camera_coords[:, 2] = z_value
    
    if is_mouthcamera and len(camera_coords) > 2:
        third_point_x = camera_coords[2, 0]
        camera_coords[:, 0] -= third_point_x

    return camera_coords

def camera2image_coordinates(camera_coords, camera_matrix_file):
    """
    カメラ座標系から画像座標系への変換を行う。
    """
    # Load the camera matrix from the npy file
    camera_matrix = np.load(camera_matrix_file)
    
    # Remove the third coordinate
    camera_coords = camera_coords[:, :2]
    
    # Add a third coordinate of 1 to the camera coordinates (homogeneous coordinates)
    ones = np.ones((camera_coords.shape[0], 1))
    camera_coords_homogeneous = np.hstack((camera_coords, ones))
    
    # Convert camera coordinates to image coordinates
    image_coords_homogeneous = np.dot(camera_matrix, camera_coords_homogeneous.T).T
    
    # Normalize the image coordinates
    image_coords = image_coords_homogeneous[:, :2] / image_coords_homogeneous[:, 2].reshape(-1, 1)
    
    return image_coords

def camera2world_coordinates(camera_coords, rotation_matrix, translation_vector):
    """
    カメラ座標系からワールド座標系への変換を行う。
    """
    # Inverse rotation matrix
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    
    # Convert camera coordinates to world coordinates
    world_coordinates = np.dot(inv_rotation_matrix, (camera_coords - translation_vector.T).T).T
    
    return world_coordinates

def world2camera_coordinates(world_coords, rotation_matrix, translation_vector):
    """
    ワールド座標系からカメラ座標系への変換を行う。
    """
    # Convert world coordinates to camera coordinates
    camera_coords = np.dot(rotation_matrix, world_coords.T).T + translation_vector.T
    
    return camera_coords

def mm2pixel(mm, dpi):
    """
    mmをpixelに変換する。
    """
    return mm * dpi / 25.4

def pixel2mm(pixel, dpi):
    """
    pixelをmmに変換する。
    """
    return pixel * 25.4 / dpi


if __name__ == "__main__":
    # テスト用のカメラ行列（3x3の行列）
    camera_matrix = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])

    # テスト用の画像座標点（N x 2の行列）
    image_points = np.array([
        [400, 300],
        [320, 240],
        [250, 200]
    ])

    # カメラ行列をファイルに保存
    camera_matrix_file = 'camera_matrix.npy'
    np.save(camera_matrix_file, camera_matrix)

    # 関数の動作をテストする
    z_value = 200  # カメラ座標系のz座標（ピクセル単位）
    camera_coordinates = image2camera_coordinates(image_points, z_value, camera_matrix_file)

    # 結果を表示
    print("Image Points:")
    print(image_points)
    print("\nCamera Coordinates:")
    print(camera_coordinates)
