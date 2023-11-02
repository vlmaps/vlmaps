import numpy as np


def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotation_matrix_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


yaw = np.radians(90)  # 90-degree rotation around the z-axis
pitch = np.radians(-90)  # -90-degree rotation around the new x-axis

T_rotation_forward = rotation_matrix_z(yaw)
T_rotation_left = rotation_matrix_x(pitch)

T_new = np.dot(T_rotation_left, T_rotation_forward)
print(T_new)
