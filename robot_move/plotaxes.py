import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_frame(ax, T, frame_length=10, label=None):
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
    
    ax.quiver(*origin, *x_axis, color='r', length=frame_length, arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color='g', length=frame_length, arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color='b', length=frame_length, arrow_length_ratio=0.1)
    
    if label:
        ax.text(*origin, label, fontsize=12)

def rotation_matrix(axis, angle):
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

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# World frame (identity transformation)
T_world = homogenous_transform()
plot_frame(ax, T_world, label='World')

# Example SE(3) transformations
# Frame 1: Translated along x and rotated about z
T1 = homogenous_transform(
    rotation=rotation_matrix('z', np.pi/4),
    position=np.array([1, 1, 0])
)
# plot_frame(ax, T1, label='Frame 1')

# Frame 2: Translated and rotated in 3D
T2 = homogenous_transform(
    rotation=rotation_matrix('y', np.pi/3) @ rotation_matrix('x', np.pi/6),
    position=np.array([1, -1, 1])
)
# plot_frame(ax, T2, label='Frame 2')

# Frame 3: Another arbitrary transformation
T3 = homogenous_transform(
    rotation=rotation_matrix('x', -np.pi/4) @ rotation_matrix('z', np.pi/3),
    position=np.array([-1, 0, 2])
)
# plot_frame(ax, T3, label='Frame 3')

# Draw a line segment between origins to show relations
# origins = np.vstack([T_world[:3, 3], T1[:3, 3], T2[:3, 3], T3[:3, 3]])
# ax.plot(origins[:,0], origins[:,1], origins[:,2], 'k--', alpha=0.5)


Mat = np.array([[ 1.24025798e-01, -4.03586167e-01, -9.06496446e-01,  3.23069000e+02],
                [ 1.69688323e-01, -8.91466482e-01,  4.20111156e-01,  3.05140000e+01],
                [-9.77662250e-01, -2.05926483e-01, -4.20809815e-02,  3.36078000e+02],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


plot_frame(ax, Mat, label='Wrist')

# Set view limits
ax.set_xlim([-600, 600])
ax.set_ylim([-600, 600])
ax.set_zlim([-100, 300])

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SE(3) Transformation Visualization')
ax.view_init(elev=20, azim=-35)

plt.tight_layout()
plt.show()
