import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urdf_parser_py.urdf import URDF
import os

def plot_frame(ax, position, rotation, length=0.1, label=None):
    """Plot a coordinate frame at given position with rotation matrix"""
    x_axis = position + rotation[:, 0] * length
    y_axis = position + rotation[:, 1] * length
    z_axis = position + rotation[:, 2] * length
    
    ax.quiver(*position, *(x_axis-position), color='r', arrow_length_ratio=0.1)
    ax.quiver(*position, *(y_axis-position), color='g', arrow_length_ratio=0.1)
    ax.quiver(*position, *(z_axis-position), color='b', arrow_length_ratio=0.1)
    
    if label:
        ax.text(*position, label, fontsize=8)

def get_link_frames(robot, joint_angles=None):
    """
    Compute link frames using forward kinematics
    Simplified version assuming all joints are revolute on their local z-axis
    """
    if joint_angles is None:
        joint_angles = {j.name: 0 for j in robot.joints if j.type == 'revolute'}
    
    # Build transform chain
    transforms = {}
    transforms[robot.get_root()] = np.eye(4)
    
    def traverse(link_name):
        link = robot.link_map[link_name]
        
        for joint in link.parent_joints:
            child_link = robot.link_map[joint.child]
            
            # Get transform from parent to joint
            if joint.origin is not None:
                xyz = joint.origin.xyz if joint.origin.xyz else [0,0,0]
                rpy = joint.origin.rpy if joint.origin.rpy else [0,0,0]
            else:
                xyz, rpy = [0,0,0], [0,0,0]
            
            # Rotation matrix from RPY angles
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                          [0, np.sin(rpy[0]), np.cos(rpy[0])]])
            Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                          [0, 1, 0],
                          [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
            Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0],
                          [np.sin(rpy[2]), np.cos(rpy[2]), 0],
                          [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            # Joint rotation if revolute
            if joint.type == 'revolute' and joint.name in joint_angles:
                angle = joint_angles[joint.name]
                joint_rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0],
                                     [0, 0, 1]])
                R = R @ joint_rot
            
            # Build 4x4 transform
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = xyz
            
            # Multiply with parent transform
            transforms[child_link.name] = transforms[link_name] @ T
            traverse(child_link.name)
    
    traverse(robot.get_root())
    return transforms

def visualize_urdf(urdf_file, joint_angles=None):
    """Visualize URDF robot in matplotlib"""
    robot = URDF.from_xml_file(urdf_file)
    transforms = get_link_frames(robot, joint_angles)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each link frame
    for link_name, T in transforms.items():
        pos = T[:3, 3]
        rot = T[:3, :3]
        plot_frame(ax, pos, rot, label=link_name)
        
    # Plot connections between links
    for joint in robot.joints:
        parent_pos = transforms[joint.parent].T[:3, 3]
        child_pos = transforms[joint.child].T[:3, 3]
        ax.plot([parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]], 'k-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'URDF Visualization: {os.path.basename(urdf_file)}')
    
    # Set equal aspect ratio
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    max_range = np.max(limits[:,1] - limits[:,0]) * 0.5
    mid_x = np.mean(limits[0,:])
    mid_y = np.mean(limits[1,:])
    mid_z = np.mean(limits[2,:])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf_file', help='Path to URDF file')
    args = parser.parse_args()
    
    # Example joint angles dictionary (modify as needed)
    joint_angles = {
        # "joint1": np.pi/4,
        # "joint2": -np.pi/6
    }
    
    visualize_urdf(args.urdf_file, joint_angles)
