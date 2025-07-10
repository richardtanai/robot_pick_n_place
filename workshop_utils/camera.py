import pyrealsense2 as rs
import numpy as np
import cv2
from .math_utils import vecs_to_se3mat

def get_dim_high(points, dim):
    if len(points.shape) ==3:
        tmp = points[:,:,dim].flatten()
    elif len(points.shape) ==2:
        tmp = points[:,dim].flatten()
    tmp = tmp[tmp!=0.0]
    tmp = np.sort(tmp)
    n = tmp[::-1]
    n_bottom = n[0:10].mean()
    return n_bottom

def get_dim_low(points, dim):
    if len(points.shape) ==3:
        tmp = points[:,:,dim].flatten()
    elif len(points.shape) ==2:
        tmp = points[:,dim].flatten()
    tmp = tmp[tmp!=0.0]
    tmp = np.sort(tmp)
    n_top = tmp[0:10].mean()
    return n_top

def get_bounds(points):
    
    x_l = get_dim_low(points,0)
    y_l = get_dim_low(points,1)
    z_l = get_dim_low(points,2)
    x_h = get_dim_high(points,0)
    y_h = get_dim_high(points,1)
    z_h = get_dim_high(points,2)
    return x_l, y_l, z_l, x_h, y_h, z_h

def draw_lines_from_bounds(img, depth_intrinsics, x_l, y_l, z_l, x_h, y_h, z_h):
    pt000 = np.array([x_l, y_l, z_l])
    pt100 = np.array([x_l, y_l, z_h])
    pt010 = np.array([x_l, y_h, z_l])
    pt110 = np.array([x_l, y_h, z_h])
    pt001 = np.array([x_h, y_l, z_l])
    pt101 = np.array([x_h, y_l, z_h])
    pt011 = np.array([x_h, y_h, z_l])
    pt111 = np.array([x_h, y_h, z_h])

    pixel000 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt000)
    pixel001 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt001)
    pixel010 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt010)
    pixel011 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt011)
    pixel100 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt100)
    pixel101 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt101)
    pixel110 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt110)
    pixel111 = rs.rs2_project_point_to_pixel(depth_intrinsics,pt111)

    pixel000 = int(pixel000[0]) , int(pixel000[1])
    pixel001 = int(pixel001[0]) , int(pixel001[1])
    pixel010 = int(pixel010[0]) , int(pixel010[1])
    pixel011 = int(pixel011[0]) , int(pixel011[1])
    pixel100 = int(pixel100[0]) , int(pixel100[1])
    pixel101 = int(pixel101[0]) , int(pixel101[1])
    pixel110 = int(pixel110[0]) , int(pixel110[1])
    pixel111 = int(pixel111[0]) , int(pixel111[1])

    top_pt = pixel000

    image_with_line = img
    image_with_line = cv2.line(image_with_line,pixel000,pixel001,(0,255,255))
    image_with_line = cv2.line(image_with_line,pixel000,pixel010,(0,255,255))
    image_with_line = cv2.line(image_with_line,pixel011,pixel001,(0,255,255))
    image_with_line = cv2.line(image_with_line,pixel011,pixel010,(0,255,255))

    image_with_line = cv2.line(image_with_line,pixel100,pixel101,(255,0,255))
    image_with_line = cv2.line(image_with_line,pixel100,pixel110,(255,0,255))
    image_with_line = cv2.line(image_with_line,pixel111,pixel101,(255,0,255))
    image_with_line = cv2.line(image_with_line,pixel111,pixel110,(255,0,255))

    image_with_line = cv2.line(image_with_line,pixel000,pixel100,(255,255,0))
    image_with_line = cv2.line(image_with_line,pixel001,pixel101,(255,255,0))
    image_with_line = cv2.line(image_with_line,pixel010,pixel110,(255,255,0))
    image_with_line = cv2.line(image_with_line,pixel011,pixel111,(255,255,0))

    return image_with_line, top_pt

class D435:
    """
    A wrapper class for the Intel Realsense D435 Camera
    """
    def __init__(self, w = 640, h = 480, fps=15, bag_path=None) -> None:
        self.w = w
        self.h = h
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        
        if bag_path is not None:
            rs.config.enable_device_from_file(config, bag_path)

        # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()


        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()

        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.pc = rs.pointcloud()

        for i in range(10):
            # dump first 60 frames
            frames = self.pipeline.wait_for_frames()


        color_stream = profile.get_stream(rs.stream.color)
        color_stream_profile = color_stream.as_video_stream_profile()
        intrinsics = color_stream_profile.get_intrinsics()
        self.distortion_coefficients = np.array([intrinsics.coeffs])
        self.calibration_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx ],
                               [0, intrinsics.fy, intrinsics.ppy],
                               [0,0,1]])


        depth_stream = profile.get_stream(rs.stream.depth)
        depth_stream_profile = depth_stream.as_video_stream_profile()

        self.depth_intrinsics = depth_stream_profile.get_intrinsics()





    def get_depth_scale(self):
        return self.depth_scale


    def capture(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            pass


        self.depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((color_image, depth_colormap))

        self.pc.map_to(color_frame)
        points = self.pc.calculate(aligned_depth_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(self.h, self.w, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(self.h, self.w, 2)  # uv

        return {"color":color_image, "depth_image": depth_image,"depth_colormap":depth_colormap, "verts": verts, "depth_intrinsics": self.depth_intrinsics, "points": points}

    def capture_color(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid

        color_image = np.asanyarray(color_frame.get_data())

        # images = np.hstack((color_image, depth_colormap))


        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', images)
        # key = cv2.waitKey(1)
        # key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()

        return color_image
    
    def detect_aruco_all(self):
        img = self.capture()
        color_image = img["color"]
        depth_colormap = img["depth_colormap"]
        #TODO
        
    
    def stop(self):
        self.pipeline.stop()

    def draw_3d_boxes(self, img, depth_intrinsics, verts, masks, draw_size=True):
        text_size = 0.5
        text_th = 1
        for i in range(len(masks)):
            img_points = verts[masks[i],:]
            x_l, y_l, z_l, x_h, y_h, z_h = get_bounds(img_points)

            img, top_pt = draw_lines_from_bounds(img, depth_intrinsics,  x_l, y_l, z_l, x_h, y_h, z_h)

            if draw_size:
                img = cv2.putText(img, f"X:{int((x_h-x_l)*1000)}mm Y:{int((y_h-y_l)*1000)}mm Z:{int((z_h-z_l)*1000)}mm", top_pt, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        
        return img
    
    def detect_aruco_T(self, marker_size= 0.040, aruco_dict_type=cv2.aruco.DICT_5X5_50):
        # aruco_dict_type = ARUCO_DICT["DICT_5X5_50"]
        # aruco_dict_type = cv2.aruco.DICT_5X5_50
        # marker_size = 0.040

        # matrix_coefficients = np.load("calibration_matrix.npy")
        # distortion_coefficients = np.load("distortion_coefficients.npy")
        cap = self.capture()

        img = cap["color"]

        matrix_coefficients = self.calibration_matrix
        distortion_coefficients = self.distortion_coefficients

        # h,w,_ = img.shape
        # width=600
        # height = int(width*(h/w))
        # image = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, arucoDict,parameters=arucoParams)

        res = {}

        if len(corners) > 0:
                for i in range(0, len(ids)):
                    # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, matrix_coefficients,
                                                                            distortion_coefficients)
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(img, corners) 

                    # Draw Axis
                    cv2.drawFrameAxes(img, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

                    temp = {}
                    temp["tvec"] = tvec
                    temp["rvec"] = rvec
                    temp["marker"] = markerPoints
                    temp["corners"] = corners[i]

                    T = vecs_to_se3mat(rvec, tvec)

                    temp["T"] = T

                    res[int(ids[i][0])] = temp

        return res, img

if __name__ == "__main__":
    myd435 = D435()
    try:
        while True:
            img = myd435.capture()
            color_image = img["color"]
            depth_colormap = img["depth_colormap"]

            images = np.hstack((color_image, depth_colormap))


            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        myd435.stop()


