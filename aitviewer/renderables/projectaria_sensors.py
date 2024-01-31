from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.scene.node import Node
from aitviewer.viewer import Viewer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.billboard import AriaBillboard
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.scene.camera import PinholeCamera, OpenCVCamera
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import (
    bisection_timestamp_search,
    filter_points_from_confidence,
    get_nearest_pose
)
from tqdm import tqdm

import projectaria_tools.core.mps as mps
import numpy as np
import argparse
import os
import cv2

class ProjectAriaSensors(CoordinateSystem):
    def __init__(
        self,
        trajectory_folder_path,
        vrs_file_path = None,
        frame_rate = 60,
        try_open_cv_camera = False,
        project_3d_points = False,
        use_undistortion_camera_for_projection = False,
        **kwargs,
    ):
        """
        Initializer for Project Aria Glasses sensors (look at Meta's project aria). The device consists of up to 5 cameras (2 eye tracking cameras, 1 RGB camera and 2 SLAM cameras).
        The device coordinate system is the coordinate frame of the left SLAM camera.
        
        TODO: Documentation
        """
        
        # Load mesh, TODO
        #glasses = trimesh.load("../examples/resources/Glasses.obj")
        #self.glasses_mesh = Meshes(
        #    glasses.vertices,
        #    name="Cube",
        #    glasses.faces,
        #    position=[7.0, 0.0, 0.0],
        #    flat_shading=True,
        #    scale=0.1
        #)
        
        # Path convention
        if vrs_file_path is not None:
            vrs_file_path = vrs_file_path
        else:
            vrs_files = []
            vrs_files += [each for each in os.listdir(trajectory_folder_path) if each.endswith('.vrs')]
            if len(vrs_files) == 0:
                raise FileNotFoundError("No vrs file found in the MPS folder. Please specify the path to the vrs file with the -V argument.")
            vrs_file_path = os.path.join(trajectory_folder_path, vrs_files[0])
        # open_loop_trajectory_path = os.path.join(trajectory_folder_path, "open_loop_trajectory.csv") # not used currently
        closed_loop_trajectory_path = os.path.join(trajectory_folder_path, "closed_loop_trajectory.csv")
        calibration_path = os.path.join(trajectory_folder_path, "online_calibration.jsonl")
        self.semidense_pointcloud_path = os.path.join(trajectory_folder_path, "semidense_points.csv.gz")
        # Check data sources
        self.__check_data_paths(closed_loop_trajectory_path, calibration_path, vrs_file_path, self.semidense_pointcloud_path)            
        
        # Init class variables
        self.renderables = [] # list of all renderables
        self.try_open_cv_camera = try_open_cv_camera
        self.camera_rgb = None
        self.show_rgb_images = True
        self.visualize_pointcloud = True 
        self.use_undistortion_camera_for_projection = use_undistortion_camera_for_projection
        self.frame_rate = frame_rate
        self.__init_trajectory(closed_loop_trajectory_path)
        self.__init_timestamps_ns()        
        self.__init_online_calibration(calibration_path)
        self.__init_sensor_data() # inits dictionary to flexibly hold several sensor calibration data
        self.__init_point_cloud()
        self.vrs_provider = data_provider.create_vrs_data_provider(vrs_filename = vrs_file_path)
        self.project_3d_points = project_3d_points
        
        # Load all necessary data to create renderable for device as well as use the data to render all sensors
        self.__init_device_pos_or()
        
        # receive all data necessary to create renderables for the sensors
        self.__prepare_all_sensory_data()
        
        # init super class
        super().__init__(name="Aria",length=1.0, icon="\u008a", rb_pos = self.device_positions_x_y_z, rb_ori = self.device_orientation_x_y_z, **kwargs)
        
        # init all renderables except rgb camera
        # RGB Renderable has to be set manually, as Viewer is required as parameter
        self.__set_glasses_mesh_renderable()
        if self.visualize_pointcloud:
            self.__set_point_cloud_renderable()
        self.__set_sensor_renderables() 
    
    def __check_data_paths(self, closed_loop_trajectory_path, calibration_path, vrs_file_path, semidense_pointcloud_path):
        # Check paths
        for p in [closed_loop_trajectory_path, calibration_path]:
            assert os.path.exists(p), "Path {} does not exist.".format(p)
        
        if not os.path.exists(vrs_file_path):
            print("Did not find vrs file. Won't be able to visualize the pictures and the exact sensors.")
            self.visualize_images = False
        
        if not os.path.exists(semidense_pointcloud_path):
            print("Did not find semi-dense pointcloud data. Won't be able to visualize the pointcloud.")
            self.visualize_pointcloud = False
            
    def __init_timestamps_ns(self):
        # create timestamps for each millisecond (but unit is nanoseconds)
        assert hasattr(self, "trajectory"), "Call __init_trajectory() before __init_timestamps_ns()"
        start_time_device_ns = int((self.trajectory[0].tracking_timestamp.total_seconds() + 1) * 1e9) # add 1 ns to avoid having no pose for the first timestamp
        end_time_device_ns = int(self.trajectory[-1].tracking_timestamp.total_seconds() * 1e9)
        fps_to_nspf = 1e9 / self.frame_rate # 1e9 nanoseconds per second, 1 / fps frames per second
        self.timestamps_ns = np.linspace(start_time_device_ns+1e6, end_time_device_ns-1e6, int((end_time_device_ns - start_time_device_ns - 2e6)/fps_to_nspf)) # Cut off at beginning and end, as there are no poses

    def __init_trajectory(self, closed_loop_trajectory_path):
        # Load closed loop trajectory
        closed_loop_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory_path)
        self.trajectory = closed_loop_trajectory
    
    def __init_online_calibration(self, calibration_path):
        # Sample the Online Calibration for each timestamp
        assert hasattr(self, "timestamps_ns"), "Call __init_timestamps_ns() before __init_online_calibration()"
        self.online_calibrations_filtered = []
        online_calibs = mps.read_online_calibration(calibration_path)
        for time_ns in self.timestamps_ns: # Online Calibration returns a list of device calibrations
            nearest_index = bisection_timestamp_search(online_calibs, time_ns)
            if nearest_index is None:
                self.online_calibrations_filtered.append(None)
            else:
                self.online_calibrations_filtered.append(online_calibs[nearest_index])
    
    def __init_sensor_data(self):
        # Get all sensory data in a dynamic dictionary
        assert hasattr(self, "timestamps_ns"), "Call __init_timestamps_ns() before __init_sensor_data()"
        assert hasattr(self, "online_calibrations_filtered"), "Call __init_online_calibration() before __init_sensor_data()"
        self.sensor_data = {}
        for cal in self.online_calibrations_filtered:
            if cal is not None:
                sample_calibration = cal
                break
        for cam in sample_calibration.camera_calibs:
            self.sensor_data[cam.get_label()] = {"positions": np.zeros((len(self.timestamps_ns), 3)),
                                            "orientations": np.repeat(np.eye(3)[np.newaxis, :], len(self.timestamps_ns), axis=0)}
        for imu in sample_calibration.imu_calibs:
            self.sensor_data[imu.get_label()] = {"positions": np.zeros((len(self.timestamps_ns), 3)),
                                            "orientations": np.repeat(np.eye(3)[np.newaxis, :], len(self.timestamps_ns), axis=0)}
    
    def __init_device_pos_or(self):
        # Get positions and orientation of the device and all sensors
        self.device_positions_x_y_z = np.zeros((len(self.timestamps_ns), 3))
        self.device_orientation_x_y_z = np.zeros((len(self.timestamps_ns), 3, 3))
        self.T_world_device = [] #SE3 format of device extrinsics
        # A trajectory is just a list of poses
        for relative_timestamp, time_ns in enumerate(self.timestamps_ns): # relative_timestamp is the index of the timestamp, describing the time since record-start, rather than the actual timestamp of the device
            pose = get_nearest_pose(self.trajectory, time_ns) # ClosedLoopTrajectoryPose or OpenLoopTrajectoryPose object that contains world coordinate frame graphUid
            if not pose: continue # Check if pose exists for that timestamp
            T_world_device = pose.transform_world_device # Sophus::SE3d object
            SE3d_matrix = T_world_device.to_matrix() # 
            device_position_x_y_z = np.array([SE3d_matrix[1][3], SE3d_matrix[2][3], SE3d_matrix[0][3]])
            
            self.T_world_device.append(T_world_device)
            self.device_positions_x_y_z[relative_timestamp] = device_position_x_y_z
            self.device_orientation_x_y_z[relative_timestamp] = np.array([SE3d_matrix[1][0:3], SE3d_matrix[2][0:3], SE3d_matrix[0][0:3]])
    
    def __init_point_cloud(self):
        points = mps.read_global_point_cloud(self.semidense_pointcloud_path)
        
        # Filter by inv depth and depth to only show points, where the algorithm is confident enough about its 3d position
        points = filter_points_from_confidence(points)
        
        # transform points from right-handed coordinate systemto ait viewer convention (left-handed)
        self.point_cloud = np.stack([np.array([x.position_world[1],x.position_world[2],x.position_world[0]]) for x in points])[np.newaxis, :] # shape (F, N, 3) with left-handed coordinate system
        self.original_point_cloud = np.stack([np.array([x.position_world[0], x.position_world[1],x.position_world[2]]) for x in points])[np.newaxis, :] # shape (F, N, 3) with right-handed coordinate system

    def __prepare_all_sensory_data(self):
        # For all sensors, prepare the data needed to create renderables
        assert hasattr(self, "device_positions_x_y_z"), "Call __init_device_pos_or() before __prepare_all_sensory_data()"
        
        # Save calibrations for each timestamp
        for relative_timestamp, time_ns in enumerate(self.timestamps_ns): # relative_timestamp is the index of the timestamp, describing the time since record-start, rather than the actual timestamp of the device
            # Get each sensor calibration from online calibrations
            online_calibration = self.online_calibrations_filtered[relative_timestamp]
            T_world_device = self.T_world_device[relative_timestamp]
            
            if online_calibration is None:
                print(f"WARNING: No online sensor calibration found for timestamp {time_ns}ns. Relative Calibration to the device from the previous timestamp will be copied.")
                for sensor in self.sensor_data.keys():
                    self.sensor_data[sensor]["positions"][relative_timestamp] = (self.sensor_data[sensor]["positions"][relative_timestamp-1])
                    self.sensor_data[sensor]["orientations"][relative_timestamp] = (self.sensor_data[sensor]["orientations"][relative_timestamp-1])
            else:
                # Define function to add calibrations to the sensor data dictionary (to be later used for rendering)
                def add_calibrations(sensor_calibration, transform_device_sensor_fun):
                    sensor_name = sensor_calibration.get_label()
                    T_device_sensor = transform_device_sensor_fun()
                    T_world_sensor = T_world_device @ T_device_sensor
                    T_world_sensor_matrix = T_world_sensor.to_matrix()
                    
                    # position and orientation
                    sensor_position_x_y_z = np.array([T_world_sensor_matrix[1][3], T_world_sensor_matrix[2][3], T_world_sensor_matrix[0][3]])
                    self.sensor_data[sensor_name]["positions"][relative_timestamp] = sensor_position_x_y_z
            
                    sensor_orientation = np.array([T_world_sensor_matrix[1][0:3], T_world_sensor_matrix[2][0:3], T_world_sensor_matrix[0][0:3]])
                    self.sensor_data[sensor_name]["orientations"][relative_timestamp] = sensor_orientation
                    
                    # Special case: To test OpenCV Camera for the rgb-camera, we need to save the intrinsics and extrinsics separately for each timestamp
                    if sensor_name == "camera-rgb":
                        
                        # permutation_matrix = np.array([[0,1,0],[0,0,1],[1,0,0]])
                        
                        self.sensor_data["camera-rgb"]["image_size"] = sensor_calibration.get_image_size()
                        
                        if self.project_3d_points:
                            # Initialize lists for extrinsics without translating original R to aitviewer convention, as during projection, the original point cloud is used
                            if "Rt_original" not in self.sensor_data["camera-rgb"].keys():
                                self.sensor_data["camera-rgb"]["Rt_original"] = []
                            # extrinsics
                            T_sensor_world = T_world_sensor.inverse().to_matrix()
                            self.sensor_data[sensor_name]["Rt_original"].append(T_sensor_world[:-1])
                            
                        if self.try_open_cv_camera:
                            # TODO: Doesn't work yet, still buggy
                            # sensor_orientation = np.array([T_world_sensor_matrix[2][0:3], -T_world_sensor_matrix[1][0:3], T_world_sensor_matrix[0][0:3]])
                            # Initialize lists for intrinsics and extrinsics
                            if "Rt" not in self.sensor_data["camera-rgb"].keys():
                                self.sensor_data["camera-rgb"]["Rt"] = []
                                self.sensor_data["camera-rgb"]["K_linear"] = []
                                self.sensor_data["camera-rgb"]["calibration"] = []
                            
                            # change extrinsics from right-handed to ait viewer convention (left-handed) + permutation in coordinates
                            T_sensor_world_matrix = T_world_sensor.inverse().to_matrix()
                            
                            R = sensor_orientation.copy().T # inverse is same as transpose for rotation matrices
                            R[1:, :] = R[1:, :].copy() * -1.0 # weird opencv convention!
                            
                            t_inv = sensor_position_x_y_z.copy()
                            t = -R @ t_inv# weird opencv convention!
                            # t_permuted =  permutation_matrix @ t_original
                            
                            # R_permuted = -np.array([[1,0,0],[0,1,0],[0,0,1]]).T
                            Rt = np.concatenate((R, t[:, np.newaxis]), axis=1)
                            # location = -Rt_permuted[:, 0:3].T @ Rt_permuted[:, 3]
                            
                            # Rt = self.current_Rt
                            # pos = -Rt[:, 0:3].T @ Rt[:, 3]
                        
                            self.sensor_data[sensor_name]["Rt"].append(Rt)
                            
                            # intrinsics
                            focal_length = sensor_calibration.get_focal_lengths()
                            principal_points = sensor_calibration.get_principal_point()
                            K = np.zeros((3,3)) # K is intrinsic matrix of focal lengths and principal points
                            K[0][0] = focal_length[1]
                            K[1][1] = focal_length[0]
                            K[0][2] = principal_points[1]
                            K[1][2] = principal_points[0]
                            K[2][2] = 1
                            self.sensor_data[sensor_name]["K_linear"].append(K)
                            
                            # full calibration
                            self.sensor_data["camera-rgb"]["calibration"].append(sensor_calibration)
                
                for camera_calib in online_calibration.camera_calibs:
                    add_calibrations(camera_calib, camera_calib.get_transform_device_camera)
                for imu_sensor in online_calibration.imu_calibs:
                    add_calibrations(imu_sensor, imu_sensor.get_transform_device_imu)
            
    def __set_glasses_mesh_renderable(self):
        return # TODO
        if self.mesh is None:
            self.mesh = Meshes(self.glasses_mesh.vertices, 
                         self.glasses_mesh.faces, 
                         self.glasses_mesh.face_normals, 
                         self.glasses_mesh.vertex_normals, 
                         self.glasses_mesh.vertex_colors, 
                         self.glasses_mesh.face_colors,
                         self.glasses_mesh.uv_coords,
                         flat_shading=True,
                         name="Aria Glasses Mesh")
        return self.mesh
    
    def __set_point_cloud_renderable(self):
        assert hasattr(self, "point_cloud"), "Call __check_data_paths() before __set_point_cloud_renderable()"
        point_cloud = PointClouds(self.point_cloud, color=(0, 0, 0, 0.8), name="point_cloud")
        self.renderables.append(point_cloud)
    
    def __set_sensor_renderables(self):
        for sensor in self.sensor_data.keys():
            if sensor != "camera-rgb":
                # render all (non rgb camera) sensors as coordinate systems
                rb_position = np.expand_dims(self.sensor_data[sensor]["positions"], 1)
                rb_orientation = self.sensor_data["camera-rgb"]["orientations"][:, np.newaxis]
                sensor_object = CoordinateSystem(rb_pos = rb_position, rb_ori = rb_orientation, length=0.05, color=(0.3, 0.3, 0.3, 1), icon="\u0086", name=sensor)
                self.renderables.append(sensor_object)
    
    def set_rgb_camera_renderable(self, viewer, visualize_billboard=True): 
        self.rgb_camera_exists = False
        for sensor in self.sensor_data.keys():
            if sensor == "camera-rgb":
                # prepare undistortion of fisheye rgb images for billboard. It is important, that the target projection matrix for undistorting images shares the focal_length /fov with the pinhole camera used for rendering
                original_image_size = self.sensor_data["camera-rgb"]["image_size"][0]
                focal_length = 1000
                self.pinhole_calibration_for_undistortion = calibration.get_linear_camera_calibration(original_image_size, original_image_size, focal_length)
                
                if self.try_open_cv_camera:
                    # TODO: OpenCVCamera - doesn't work yet
                    K = np.array(self.sensor_data["camera-rgb"]["K_linear"])
                    Rt = np.array(self.sensor_data["camera-rgb"]["Rt"])
                    self.camera_rgb = OpenCVCamera(K=K, Rt=Rt, cols=viewer.window_size[0], rows=viewer.window_size[1], viewer=viewer)
                    # print("Open_CV rotation \n", self.camera_rgb.current_rotation)
                else:
                    # TODO: when doing projections of the 3d points on the billboard's image, the projections using the Pinhole Camera's view-projection matrix is slightly off
                    # That MAY be (not 100% sure) because the PinholeCamera's fov is calculated by the focal length of the undistortion camera, which is slightly different from the original fisheye camera, where the rgb images are taken from
                    
                    # Pinhole Camera: use focal length from undistortion to calculate fov
                    targets = self.sensor_data["camera-rgb"]["positions"] + self.sensor_data["camera-rgb"]["orientations"] @ np.array([0,0,1])
                    fov = np.rad2deg(2 * np.arctan(original_image_size / (2 * focal_length)))# get fov of distorting camera --> fov = 2 * arctan(height/2 * focal_length)
                    self.camera_rgb = PinholeCamera(self.sensor_data["camera-rgb"]["positions"], targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer, fov=fov, name=sensor)

                self.renderables.append(self.camera_rgb)
                self.rgb_camera_exists = True
                
        if not self.rgb_camera_exists:
            print("Warning: RGB Camera Sensor do not exist in this recording. Camera and Billboard will not be rendered.")
            
        if visualize_billboard and self.rgb_camera_exists:
            self.__set_rgb_billboard_renderable()
    
    def __set_rgb_billboard_renderable(self):
        
        original_image_size = self.sensor_data["camera-rgb"]["image_size"][0]
       
        if not self.try_open_cv_camera: # pinhole camera
            c1, c2 = self.pinhole_calibration_for_undistortion.get_principal_point() # camera center from undistorting linear camera
            f1, f2 = self.pinhole_calibration_for_undistortion.get_focal_lengths() # focal length from undistorting linear camera
            self.camera_rgb.update_matrices(width=original_image_size, height=original_image_size)
        else: # opencv camera
            self.camera_rgb.update_matrices(width=original_image_size, height=original_image_size)
        
        if self.project_3d_points:
            
            # color of the projected points
            color = (150, 0, 0)
            
            # For undistortion: Get proejction matrix from first fisheye calibration (only intrinsics are needed for the undistortion)
            camera_rgb_local_calibration = self.vrs_provider.get_device_calibration().get_camera_calib("camera-rgb")
            
            # define function to undistort and rotate the image during rendering and project 3D points onto the image
            def process_image(raw_image, current_frame_id):
                # undistort and rotate 90 degrees clockwise from the original fisheye (which is 90 degrees rotated counterclockwise)
                undistorted_image = calibration.distort_by_calibration(raw_image, self.pinhole_calibration_for_undistortion, camera_rgb_local_calibration)
                img = np.rot90(undistorted_image, k=3)
                
                # get projection parameters and project 3d coordinates to pixel coordinates
                if self.try_open_cv_camera: 
                    # opencv camera 
                    # TODO: Note: Cannot be debugged before the general opencv pipeline is working. Except for some result flips, shouldnt be too buggy
                    K = self.camera_rgb.current_K
                    Rt = self.camera_rgb.current_Rt
                    # print("Here new position", self.camera_rgb.position)
                    
                    points3D = self.point_cloud[0]
                    
                    # homogenous 3D points in world coordinate system
                    points3D_homogeneous = np.insert(points3D, 3, np.array([1.0]), axis = 1).transpose() # (4, N)
                    
                    # project all points into camera coordinate system of the original rgb camera (fisheye)
                    world_to_camera_coords = Rt @ points3D_homogeneous # (3, 4) * (4, N) = (3, N)
                    
                    # project all points into image coordinate system using the linear model used to undistort the image
                    camera_to_image_coords = K @ world_to_camera_coords # (N, 3)
                    
                    # divide by z to transform from homogenous to x,y image coordinates
                    filter_index_z = camera_to_image_coords[2, :] >= 0 # positive z lie behind the camera and z cannot be 0 (seldom case, as Slam-predicted 3D points shouldn't be in camera center to begin with)
                    image_coords_checked_z = camera_to_image_coords[:,filter_index_z] # (3, N)
                    image_coords = image_coords_checked_z / image_coords_checked_z[2] # (3, N)
                    image_coords_x = image_coords[0]
                    image_coords_y = image_coords[1]
                    
                elif self.use_undistortion_camera_for_projection: 
                    # use original Rt and intrinsics from Aria undistortion camera used to undistort the rgb images from the fisheye view
                    K = np.array([[f1, 0, c1], [0, f2, c2], [0,0,1]]) # intrinsics
                    Rt = self.sensor_data["camera-rgb"]["Rt_original"][current_frame_id] # extrinsics from original rgb camera (fisheye)
                    points3D_original = self.original_point_cloud[0] # (N, 3)
                    
                    # homogenous 3D points in world coordinate system
                    points3D_homogeneous = np.insert(points3D_original, 3, np.array([1.0]), axis = 1).transpose() # (4, N)
                    
                    # project all points into camera coordinate system of the original rgb camera (fisheye)
                    world_to_camera_coords = Rt @ points3D_homogeneous # (3, 4) * (4, N) = (3, N)
                    
                    # project all points into image coordinate system using the linear model used to undistort the image
                    camera_to_image_coords = K @ world_to_camera_coords # (N, 3)
                    
                    # divide by z to transform from homogenous to x,y coordinates
                    filter_index_z = camera_to_image_coords[2, :] >= 0 # positive z lie behind the camera and z cannot be 0 (seldom case, as Slam-predicted 3D points shouldn't be in camera center to begin with)
                    image_coords_checked_z = camera_to_image_coords[:,filter_index_z] # (3, N)
                    image_coords = image_coords_checked_z / image_coords_checked_z[2] # (3, N)
                    image_coords_x_unrotated = image_coords[0]
                    image_coords_y_unrotated = image_coords[1]
                    
                    # rotate projections 90 degrees clockwise as original image was rotated too and discard the depth info image_coords_checked_z[2]
                    image_coords_x = original_image_size - image_coords_y_unrotated
                    image_coords_y = image_coords_x_unrotated
                    
                else:
                    # use AIT pinhole camera for projection
                    points3D_original = self.point_cloud[0] # (N, 3) TODO: revert
                
                    # homogenous 3D points in world coordinate system
                    points3D_homogeneous = np.insert(points3D_original, 3, np.array([1.0]), axis = 1).transpose() # (4, N)

                    # project 3D points to ndc
                    self.camera_rgb.update_matrices(width=original_image_size, height=original_image_size)
                    view_projection_matrix = self.camera_rgb.get_view_projection_matrix() # (4,4)
                    camera_to_image_coords = view_projection_matrix @ points3D_homogeneous # (4, 4) * (4, N) = (4, N)
                    
                    # divide by z to transform from homogenous to x,y coordinates
                    filter_index_z = camera_to_image_coords[3, :] >= 0 # z cannot be 0 and positive z lie behind the camera
                    image_coords_checked_z = camera_to_image_coords[:,filter_index_z] # (4, N)
                    image_coords = image_coords_checked_z / image_coords_checked_z[2] # (4, N)
                    
                    # ndc to image pixel coordinates
                    image_coords_x_unmirrored = (image_coords[0] + 1) * original_image_size / 2
                    image_coords_y_unmirrored = (image_coords[1] + 1) * original_image_size / 2
                    
                    image_coords_x = image_coords_x_unmirrored
                    image_coords_y = original_image_size - image_coords_y_unmirrored # the points are mirrored along the image height
                
                # filter points that lie outside of the image
                filter_index_x = np.logical_and(image_coords_x >= 0, image_coords_x < original_image_size)
                filter_index_y = np.logical_and(image_coords_y >= 0, image_coords_y < original_image_size)
                filter_index = np.logical_and(filter_index_x, filter_index_y)
                point_image_filtered_x = image_coords_x[filter_index]
                point_image_filtered_y = image_coords_y[filter_index]
                
                # draw projections onto the image
                radius = img.shape[0] / 1000
                downsampling_factor = original_image_size / img.shape[0] # should always be 1 but just in case somebody downsamples the images for faster rendering
                print(f"...Drawing {len(point_image_filtered_x)} points onto the image - please be patient while I work <3")
                for idx, x in enumerate(point_image_filtered_x):
                    y = point_image_filtered_y[idx]
                    point_center = (np.array([x,y])/downsampling_factor).astype(int)
                    img = cv2.circle(img.copy(), point_center, int(radius), color, 3)
                    # if idx > 100000:
                    #     break
                return img
        
        else:
            # no projection wanted; define function to undistort and rotate 90 degrees during rendering
            def process_image(raw_image, _):
                undistorted_image = calibration.distort_by_calibration(raw_image, self.pinhole_calibration_for_undistortion, camera_rgb_local_calibration)
                processed_image = np.rot90(undistorted_image, k=3)
                return processed_image
        
        # AriaBillboard, that loads each frame during rendering via the Aria vrs data provider
        billboard = AriaBillboard.from_camera_and_distance(self.vrs_provider, self.timestamps_ns, self.camera_rgb, 1.34, original_image_size, original_image_size, np.zeros(len(self.sensor_data["camera-rgb"]["positions"])),
                                                    image_process_fn = process_image)
        # billboard = AriaBillboard.from_camera_and_distance(vrs_provider, timestamps_ns, camera_rgb, 1.34, original_image_size[1], original_image_size[0], np.zeros(len(targets)-1), # TODO: weirdly the opencv camera takes a timestamp less than given, to be investigated..
        #                                            image_process_fn = process_image)
        billboard.texture_alpha = 0.9
        self.renderables.append(billboard)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-T', type=str, required=False, default="/Users/annel/Documents/Github Repositories/aria_ait/data/Profile2durationSlam_Trajectory", help="MPS folder name inside data_source path. Make sure to move the vrs Files into the folder")
    parser.add_argument('-F', type=int, required=False, default=60, help="Frame Rate to sample all data with, also used for rendering.")
    parser.add_argument('-V', type=str, required=False, default=None, help="VRS file path. If not given, the script will look for a vrs file in the MPS folder.")
    parser.add_argument('-O', type=bool, required=False, default=False, help="Test case for opencv camera")
    parser.add_argument('-P', type=bool, required=False, default=True, help="Project 3D points onto 2D image plane of the billboard.")
    parser.add_argument('-D', type=bool, required=False, default=True, help="Only relevant if P=True and O=False: Use linear undistortion camera for projection.")
    args, _ = parser.parse_known_args()
    args = dict(map(lambda arg: (arg, getattr(args, arg)), vars(args)))
    
    aria_mesh = ProjectAriaSensors(
        trajectory_folder_path=args["T"],
        vrs_file_path=args["V"],
        frame_rate=args["F"],
        try_open_cv_camera=args["O"],
        project_3d_points=args["P"],
        use_undistortion_camera_for_projection=args["D"]
    )
    
    C.update_conf({"playback_fps": args["F"]})
    C.update_conf({"scene_fps": args["F"]})
    
    v = Viewer()
    aria_mesh.set_rgb_camera_renderable(viewer=v, visualize_billboard = True)
    
    v.scene.add(*aria_mesh.renderables)
    v.run()