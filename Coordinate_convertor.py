import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def image2world_coordinates(img_points, inv_camera_matrix, rotation_matrix, tvec):
    """
    画像座標系からワールド座標系への変換を行う。cv2.calibrateCamera関数で得られるrvecs, tvecsを使用する。
    
    """
    #inv_camera_matrix = np.linalg.inv(camera_matrix)
    row = img_points.shape[0]
    img_points_homogeneous = np.hstack((img_points, np.ones((row, 1))))
    inv_rotation_matrix = np.linalg.inv(rotation_matrix)
    camera_points = np.dot(inv_camera_matrix, img_points_homogeneous.T).T
    world_points = np.dot(inv_rotation_matrix, (camera_points - tvec.T).T).T
    #print(world_points)
    return world_points

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
    lmk_2d = np.load("AnnotatedData/Nakabayashi_Annotated/NPYs/lefteye/test0_0_annotated.npy")
    camera_mtx = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_mtx.npy")
    dist = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_dist.npy")
    rvecs = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_rvecs.npy")
    tvecs = np.load("CameraCalibration/Parameters/ChessBoard_eye_left_tvecs.npy")
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
    tvec = tvecs[0]
    inv_camera_mtx = np.linalg.inv(camera_mtx)
    world_point = image2world_coordinates(lmk_2d, inv_camera_mtx, rotation_matrix, tvec)
    # 3Dプロットを作成します
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # カメラ1のポイントをプロット
    ax.scatter(world_point[:, 0], world_point[:, 1], world_point[:, 2], c='b', marker='o', label='World Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1

    # Set equal scaling
    scaling = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    # プロットデータの範囲を計算
    x_limits = (np.min(world_point[:, 0]), np.max(world_point[:, 0]))
    y_limits = (np.min(world_point[:, 1]), np.max(world_point[:, 1]))
    z_limits = (np.min(world_point[:, 2]), np.max(world_point[:, 2]))

    # 各軸の表示範囲を設定
    ax.set_xlim(x_limits[0] - 0.1 * np.ptp(x_limits), x_limits[1] + 0.1 * np.ptp(x_limits))
    ax.set_ylim(y_limits[0] - 0.1 * np.ptp(y_limits), y_limits[1] + 0.1 * np.ptp(y_limits))
    ax.set_zlim(z_limits[0] - 0.1 * np.ptp(z_limits), z_limits[1] + 0.1 * np.ptp(z_limits))
    plt.show()

    # 2Dプロットを作成します
    plt.figure()
    plt.scatter(lmk_2d[:, 0], -lmk_2d[:, 1], c='b', marker='o')  # y軸を反転
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Projection of 3D Points (Viewed from Negative Z-axis)')
    plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio for 2D plot
    plt.show()


