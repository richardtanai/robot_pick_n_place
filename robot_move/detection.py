import open3d as o3d

import workshop_utils.AR4_api as AR4_api
import time
import math
import pickle
from workshop_utils.camera import D435
import numpy as np
import matplotlib.pyplot as plt
from workshop_utils import *

myD435 = D435()


# Start streaming

depth_intrinsics = myD435.depth_intrinsics

fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

print("Intrinsics:", fx, fy, cx, cy)

try:
    while True:
        # ---- 2. Get frames ----
        res = myD435.capture()
        color_frame = res["color"]
        depth_frame = res["depth_image"]

        if depth_frame.any() or color_frame.any():
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * myD435.depth_scale  # meters
        color_image = np.asanyarray(color_frame.get_data())

        # ---- 3. Near Mask ----
        near_mask = (depth_image > 0.2) & (depth_image < 1.2)

        # ---- 4. Create Point Cloud ----
        idxs = np.argwhere(near_mask)
        if idxs.size == 0:
            continue

        z = depth_image[idxs[:,0], idxs[:,1]]
        u = idxs[:,1]
        v = idxs[:,0]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x,y,z), axis=1)
        colors = color_image[v,u]/255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # ---- 5. RANSAC Plane Segmentation ----
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                                 ransac_n=3,
                                                 num_iterations=500)
        [a,b,c,d] = plane_model
        print("Plane: {:.3f}x + {:.3f}y + {:.3f}z + {:.3f}".format(a,b,c,d))

        # ---- 6. Mask table points ----
        inlier_points = np.asarray(pcd.points)[inliers]
        table_mask_2d = np.zeros(depth_image.shape, dtype=np.uint8)
        for p in inlier_points:
            u_proj = int((p[0]*fx/p[2]) + cx)
            v_proj = int((p[1]*fy/p[2]) + cy)
            if 0<=u_proj<depth_image.shape[1] and 0<=v_proj<depth_image.shape[0]:
                table_mask_2d[v_proj,u_proj] = 255

        # ---- 7. Foreground Mask ----
        foreground_mask = cv2.bitwise_and(near_mask.astype(np.uint8)*255,
                                          cv2.bitwise_not(table_mask_2d))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        # ---- 8. Contour Detection ----
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = color_image.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(output_image, [box], 0, (0,255,0), 2)
            cv2.putText(output_image, "Candidate", (box[0][0], box[0][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ---- 9. Show Results ----
        cv2.imshow('RGB', color_image)
        cv2.imshow('Foreground Mask', foreground_mask)
        cv2.imshow('Detected Objects', output_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

finally:
    myD435.stop()
    cv2.destroyAllWindows()
