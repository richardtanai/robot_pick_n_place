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
    R = rotation_matrix(np.deg2rad(rz),np.deg2rad(ry), np.deg2rad(rx))
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

    return (p[0]*1000, p[1]*1000, p[2]*1000, gamma*180/np.pi, beta*180/np.pi ,alpha*180/np.pi) ## convert to mm and deg


def se3_to_xyz_rzyx_stable(T, eps=1e-6):
    """
    Convert SE(3) to position and ZYX Euler angles (yaw-pitch-roll), stable against gimbal lock.
    
    Parameters:
        T : (4,4) array - SE(3) transform
        eps : float - threshold for gimbal lock detection

    Returns:
        pos : (3,) position vector
        angles : (3,) list [yaw, pitch, roll] in radians
    """
    R = T[:3,:3]
    t = T[:3,3]

    # Clamp values to avoid NaN from arccos etc.
    r20 = np.clip(R[2,0], -1.0, 1.0)

    # Pitch = arcsin(-r20), but we use atan2 version
    sy = np.sqrt(R[2,1]**2 + R[2,2]**2)

    singular = sy < eps  # Gimbal lock near ±90 deg pitch

    if not singular:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:
        # Gimbal lock: pitch is ±90°, set roll = 0, solve for yaw
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)  # still safe here
        roll  = 0.0  # cannot be determined uniquely

    return (t[0], t[1], t[2], yaw, pitch, roll)



def se3_to_xyz_rzyx_stable_mm(T, eps=1e-6):
    """
    Convert SE(3) to position and ZYX Euler angles (yaw-pitch-roll), stable against gimbal lock.
    
    Parameters:
        T : (4,4) array - SE(3) transform
        eps : float - threshold for gimbal lock detection

    Returns:
        pos : (3,) position vector
        angles : (3,) list [yaw, pitch, roll] in radians
    """
    R = T[:3,:3]
    t = T[:3,3]

    # Clamp values to avoid NaN from arccos etc.
    r20 = np.clip(R[2,0], -1.0, 1.0)

    # Pitch = arcsin(-r20), but we use atan2 version
    sy = np.sqrt(R[2,1]**2 + R[2,2]**2)

    singular = sy < eps  # Gimbal lock near ±90 deg pitch

    if not singular:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:
        # Gimbal lock: pitch is ±90°, set roll = 0, solve for yaw
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)  # still safe here
        roll  = 0.0  # cannot be determined uniquely

    return (t[0]*1000, t[1]*1000, t[2]*1000, yaw*180/math.pi, pitch*180/math.pi, roll*180/math.pi)


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
    

def se3_to_xyz_rzyx_close(T, robot, eps=1e-6):
    
    Rm = T[:3,:3]
    p = T[0:3,3]


    sy = np.sqrt(Rm[2,1]**2 + Rm[2,2]**2)
    singular = sy < eps

    if not singular:
        yaw1   = np.arctan2(Rm[1,0], Rm[0,0])
        pitch1 = np.arctan2(-Rm[2,0], sy)
        roll1  = np.arctan2(Rm[2,1], Rm[2,2])

        yaw2   = (yaw1 + np.pi) % (2*np.pi)
        pitch2 = -pitch1
        roll2  = (-roll1) % (2*np.pi)
    else:
        if Rm[2,0] <= -1 + eps:
            pitch1 = np.pi/2
        else:
            pitch1 = -np.pi/2

        yaw1 = np.arctan2(-Rm[0,1], Rm[1,1])
        roll1 = 0.0

        yaw2 = yaw1
        pitch2 = pitch1
        roll2 = 0.0

    candidate1 = [yaw1, pitch1, roll1]
    candidate2 = [yaw2, pitch2, roll2]

    angles = robot.get_closer_euler(candidate1, candidate2)

    return p, angles


