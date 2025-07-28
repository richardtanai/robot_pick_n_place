import torch
import numpy as np
import cv2
import pyrealsense2 as rs
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---- Load SAM Model ----
sam_checkpoint = r"C:\Users\Richard\Desktop\robot_demo\robot_pick_n_place\robot_move\sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_batch=32,   # smaller batch for less VRAM
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92
)

# ---- Configure RealSense ----
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Get intrinsics for 3D projection
profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
cx, cy = depth_intrinsics.ppx, depth_intrinsics.ppy

print("Camera intrinsics:", fx, fy, cx, cy)

try:
    while True:
        # ---- Capture Frame ----
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # meters
        color_image = np.asanyarray(color_frame.get_data())

        # Convert BGR to RGB for SAM
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # ---- Run SAM Mask Generation ----
        masks = mask_generator.generate(rgb_image)

        output = color_image.copy()
        detected_any = False

        for mask in masks:
            m = mask['segmentation'].astype(np.uint8)*255

            # Contours
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:  # ignore tiny
                    continue

                rect = cv2.minAreaRect(cnt)
                width = min(rect[1])
                height = max(rect[1])
                if width == 0:
                    continue
                aspect_ratio = height / width

                # Detect elongated objects
                if aspect_ratio > 4:
                    detected_any = True
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)

                    # Draw box
                    cv2.drawContours(output, [box], 0, (0,255,0), 2)

                    # Create mask ROI
                    mask_roi = m > 0

                    # Extract depth values within mask
                    mask_depths = depth_image[mask_roi]
                    valid_depths = mask_depths[(mask_depths > 0.1) & (mask_depths < 1.2)]
                    if valid_depths.size == 0:
                        continue

                    median_depth = np.median(valid_depths)

                    # Get center point
                    M = cv2.moments(m)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Project to 3D
                    Z = median_depth
                    X = (cX - cx) * Z / fx
                    Y = (cY - cy) * Z / fy

                    # Draw info
                    cv2.circle(output, (cX, cY), 4, (0,0,255), -1)
                    cv2.putText(output, f"Pen candidate AR={aspect_ratio:.1f}",
                                (box[0][0], box[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.putText(output, f"X={X:.3f} Y={Y:.3f} Z={Z:.3f}",
                                (cX+5, cY-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    print("3D position:", (X,Y,Z))

        if not detected_any:
            cv2.putText(output, "No elongated objects detected.",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        # ---- Display ----
        cv2.imshow("SAM + RealSense Pen Detection", output)
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
