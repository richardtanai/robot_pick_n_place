import cv2
import numpy as np
import math

def vecs_to_se3mat(rvec,tvec):
    T = np.eye(4,4)
    T[:3,:3], _ = cv2.Rodrigues(rvec)
    T[:3,3] = tvec
    return T


def rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotation_x(theta):
    return np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

def rotation_matrix(rz, ry, rx):
       Rz = rotation_z(rz)
       Ry = rotation_y(ry)
       Rx = rotation_x(rx)
       return (Rz @ Ry) @ Rx  

def xyz_zyx_to_se3(x, y, z, rz, ry, rx):
    '''
    angles in deg
    '''
    R = rotation_matrix(rz*math.pi/180, ry*math.pi/180, rx*math.pi/180)
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])
    # Combine rotation and translation
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = np.array([x, y, z])
    return se3



def plot_frame(ax, T, frame_length=1.0, label=None):
    """
    Plot a coordinate frame in 3D
    Parameters:
        ax: matplotlib 3D axis
        T: 4x4 transformation matrix
        frame_length: length of axis lines
        label: name of the frame
    """
    origin = T[:3, 3]
    x_axis = T[:3, :3] @ np.array([frame_length, 0, 0])
    y_axis = T[:3, :3] @ np.array([0, frame_length, 0])
    z_axis = T[:3, :3] @ np.array([0, 0, frame_length])
    
    ax.quiver(*origin, *x_axis, color='r', length=frame_length, arrow_length_ratio=10)
    ax.quiver(*origin, *y_axis, color='g', length=frame_length, arrow_length_ratio=10)
    ax.quiver(*origin, *z_axis, color='b', length=frame_length, arrow_length_ratio=10)
    
    if label:
        ax.text(*origin, label, fontsize=12)

def homogenous_transform(rotation=None, position=None):
    """
    Create a 4x4 SE(3) transformation matrix
    Parameters:
        rotation: 3x3 rotation matrix (default: identity)
        position: 3D translation vector (default: zero)
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if position is not None:
        T[:3, 3] = position
    return T


def se3_to_xyz_rzyx_mm(T): ## in mm
    """
    Converts SE(3) matrix to (x, y, z, Rz, Ry, Rx) in radians.
    
    Returns:
        tuple: (x, y, z, Rz, Ry, Rx) mm and degrees
    """
    R = T[0:3,0:3]
    p = T[0:3,3]

    # Compute Euler angles ZYX
    sy = -R[2,0]
    beta = np.arcsin(sy)
    cos_beta = np.cos(beta)

    if abs(cos_beta) > 1e-6:
        alpha = np.arctan2(R[2,1]/cos_beta, R[2,2]/cos_beta)
        gamma = np.arctan2(R[1,0]/cos_beta, R[0,0]/cos_beta)
    else:
        # Gimbal lock
        alpha = 0
        gamma = np.arctan2(-R[0,1], R[1,1])

    return (p[0]*1000, p[1]*1000, p[2]*1000, gamma*180/math.pi, beta*180/math.pi ,alpha*180/math.pi) ## convert to mm and deg

def se3_to_xyz_rzyx(T): 
    """
    Converts SE(3) matrix to (x, y, z, Rz, Ry, Rx) in meters and radians.
    
    Returns:
        tuple: (x, y, z, Rz, Ry, Rx) meters and radians
    """
    R = T[0:3,0:3]
    p = T[0:3,3]

    # Compute Euler angles ZYX
    sy = -R[2,0]
    beta = np.arcsin(sy)
    cos_beta = np.cos(beta)

    if abs(cos_beta) > 1e-6:
        alpha = np.arctan2(R[2,1]/cos_beta, R[2,2]/cos_beta)
        gamma = np.arctan2(R[1,0]/cos_beta, R[0,0]/cos_beta)
    else:
        # Gimbal lock
        alpha = 0
        gamma = np.arctan2(-R[0,1], R[1,1])

    return (p[0], p[1], p[2], gamma, beta, alpha) ## convert to mm and deg

def rotation_matrix_a(axis, angle):
    """
    Create a 3x3 rotation matrix about an axis by an angle
    Parameters:
        axis: 'x', 'y', or 'z'
        angle: rotation angle in radians
    Returns:
        3x3 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])