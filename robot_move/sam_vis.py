import torch
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---- Load SAM Model ----
sam_checkpoint = r"C:\Users\Richard\Desktop\robot\realsense_workshop\robot_move\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_batch=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92
)

# ---- RealSense Setup ----
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

# ---- Table height assumption ----
TABLE_Z = 0.75
MIN_ABOVE_TABLE = 0.02

# ---- Open3D Visualizer ----
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Pen Detection")

pcd = o3d.geometry.PointCloud()
geom_added = False

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Generate masks
        masks = mask_generator.generate(rgb_image)

        # Convert depth to point cloud
        v, u = np.indices(depth_image.shape)
        Z = depth_image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points = np.stack((X, Y, Z), axis=-1).reshape(-1,3)
        colors = rgb_image.reshape(-1,3) / 255.0

        all_pen_points = []
        all_boxes = []

        for mask in masks:
            m = mask['segmentation'].astype(np.uint8)*255
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                rect = cv2.minAreaRect(cnt)
                width = min(rect[1])
                height = max(rect[1])
                if width == 0:
                    continue
                aspect_ratio = height / width
                if aspect_ratio < 4:
                    continue

                mask_idx = np.where(m.flatten()>0)[0]
                mask_zs = Z.flatten()[mask_idx]
                valid_z = mask_zs[(mask_zs > 0.1) & (mask_zs < 1.2)]
                if valid_z.size == 0:
                    continue
                median_z = np.median(valid_z)
                if median_z < (TABLE_Z + MIN_ABOVE_TABLE):
                    continue

                pen_pts = points[mask_idx]
                all_pen_points.append(pen_pts)

                # Compute bounding box
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pen_pts))
                obb.color = (1,0,0)
                all_boxes.append(obb)

        # Update main point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if not geom_added:
            vis.add_geometry(pcd)
            for b in all_boxes:
                vis.add_geometry(b)
            geom_added = True
        else:
            vis.update_geometry(pcd)
            for b in all_boxes:
                vis.add_geometry(b)

        vis.poll_events()
        vis.update_renderer()

        # Clear old boxes before next frame
        for b in all_boxes:
            vis.remove_geometry(b, reset_bounding_box=False)

finally:
    pipeline.stop()
    vis.destroy_window()
