from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.coordinate_system import CoordinateSystem
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

import projectaria_tools.core.mps as mps
import numpy as np
import trimesh
import argparse
import os

class ProjectAriaSensors(CoordinateSystem):
    def __init__(
        self,
        trajectory_folder_path,
        vrs_file_path = None,
        closed_trajectory = True,
        frame_rate = 60,
        try_open_cv_camera=False,
        **kwargs,
    ):
        """
        Initializer for Project Aria Glasses sensors (look at Meta's project aria). The device consists of up to 5 cameras (2 eye tracking cameras, 1 RGB camera and 2 SLAM cameras).
        The device coordinate system is the coordinate frame of the left SLAM camera.
        
        :param pos_over_time: A np array of shape (F, 3) containing the rigid-body centers over F time steps.
        :param rb_ori: A np array of shape (F, N, 3, 3) containing N rigid-body orientations over F time steps.
        :param radius: Radius of the sphere at the origin of the rigid body.
        :param length: Length of arrows representing the orientation of the rigid body.
        :param radius_cylinder: Radius of the cylinder representing the orientation, default is length / 50
        :param color: Color of the rigid body centers (4-tuple).
        """
        
        # Load mesh, TODO
        glasses = trimesh.load("../examples/resources/Glasses.obj")
        self.glasses_mesh = Meshes(
            glasses.vertices,
            glasses.faces,
            name="Cube",
            position=[7.0, 0.0, 0.0],
            flat_shading=True,
            scale=0.1
        )
        
        # Path convention
        if vrs_file_path is not None:
            vrs_file_path = vrs_file_path
        else:
            vrs_files = []
            vrs_files += [each for each in os.listdir(trajectory_folder_path) if each.endswith('.vrs')]
            if len(vrs_files) == 0:
                raise FileNotFoundError("No vrs file found in the MPS folder. Please specify the path to the vrs file with the -V argument.")
            vrs_file_path = os.path.join(trajectory_folder_path, vrs_files[0])
        open_loop_trajectory_path = os.path.join(trajectory_folder_path, "open_loop_trajectory.csv")
        closed_loop_trajectory_path = os.path.join(trajectory_folder_path, "closed_loop_trajectory.csv")
        calibration_path = os.path.join(trajectory_folder_path, "online_calibration.jsonl")
        self.semidense_pointcloud_path = os.path.join(trajectory_folder_path, "semidense_points.csv.gz")
        # Check data sources
        self.__check_data_paths(open_loop_trajectory_path, closed_loop_trajectory_path, calibration_path, vrs_file_path, self.semidense_pointcloud_path)            
        
        # Init class variables
        self.try_open_cv_camera = try_open_cv_camera
        self.rgb_camera = None
        self.show_rgb_images = True
        self.visualize_pointcloud = True 
        self.frame_rate = frame_rate
        self.__init_trajectory(closed_trajectory, open_loop_trajectory_path, closed_loop_trajectory_path)
        self.__init_timestamps_ns()        
        self.__init_online_calibration(calibration_path)
        self.__init_sensor_data() # inits dictionary to flexibly hold several sensor calibration data
        self.vrs_provider = data_provider.create_vrs_data_provider(vrs_filename = vrs_file_path)
        
        # Load all necessary data to create renderable for device as well as use the data to render all sensors
        self.__init_device_pos_or()
        
        # receive all data necessary to create renderables for the sensors
        self.__prepare_all_sensory_data()
        
        # init all renderables
        self.mesh = None
        
        super().__init__(name="Aria Origin",length=1.0, icon="\u008a", rb_pos = self.device_positions_x_y_z, rb_ori = self.device_orientation_x_y_z, **kwargs)
    
    def __check_data_paths(self, open_loop_trajectory_path, closed_loop_trajectory_path, calibration_path, vrs_file_path, semidense_pointcloud_path):
        # Check paths
        for p in [open_loop_trajectory_path, closed_loop_trajectory_path, calibration_path]:
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

    def __init_trajectory(self, closed_trajectory, open_loop_trajectory_path, closed_loop_trajectory_path):
        # Load open or closed loop trajectory
        if closed_trajectory:
            closed_loop_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory_path)
            self.trajectory = closed_loop_trajectory
        else:
            open_loop_trajectory = mps.read_open_loop_trajectory(open_loop_trajectory_path)
            self.trajectory  = open_loop_trajectory
    
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
                        self.sensor_data["camera-rgb"]["image_size"] = sensor_calibration.get_image_size()
                        if self.try_open_cv_camera:
                            # Initialize lists for intrinsics and extrinsics
                            if "Rt" not in self.sensor_data["camera-rgb"].keys():
                                self.sensor_data["camera-rgb"]["Rt"] = []
                                self.sensor_data["camera-rgb"]["K"] = []
                            # extrinsics
                            Rt_original = T_world_sensor_matrix[:-1]
                            Rt_permuted = Rt_original[:, [1,2,0,3]] # permute cols of R to match aitviewer convention
                            t_original = Rt_permuted[:, 3].copy()
                            Rt_permuted[:, 3] = np.array([t_original[1],t_original[2],t_original[0]]) # permute rows of t
                            self.sensor_data[sensor_name]["Rt"].append(Rt_permuted)
                            
                            # intrinsics
                            focal_length = sensor_calibration.get_focal_lengths()
                            principal_points = sensor_calibration.get_principal_point()
                            K = np.zeros((3,3)) # K is intrinsic matrix of focal lengths and principal points
                            K[0][0] = focal_length[1]
                            K[1][1] = focal_length[0]
                            K[0][2] = principal_points[1]
                            K[1][2] = principal_points[0]
                            K[2][2] = 1
                            self.sensor_data[sensor_name]["K"].append(K)   
                
                for camera_calib in online_calibration.camera_calibs:
                    add_calibrations(camera_calib, camera_calib.get_transform_device_camera)
                for imu_sensor in online_calibration.imu_calibs:
                    add_calibrations(imu_sensor, imu_sensor.get_transform_device_imu)
            
    def get_glasses_mesh_renderable(self):
        return # not done yet
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
    
    def get_point_cloud_renderable(self):
        points = mps.read_global_point_cloud(self.semidense_pointcloud_path)
        points = filter_points_from_confidence(points) # Filter by inv depth and depth to only show points, where the algorithm is confident enough about its 3d position
        point_cloud_np = np.stack([np.array([x.position_world[1],x.position_world[2],x.position_world[0]]) for x in points])[np.newaxis, :] # shape (F, N, 3) with y, z, x
        point_cloud = PointClouds(point_cloud_np, color=(0, 0, 0, 0.8), name="point_cloud")
        return point_cloud
    
    def get_sensor_renderables(self, viewer):
        renderables = {}
        for sensor in self.sensor_data.keys():
            if sensor != "camera-rgb":
                # render all (non rgb camera) sensors as coordinate systems
                rb_position = np.expand_dims(self.sensor_data[sensor]["positions"], 1)
                rb_orientation = self.sensor_data["camera-rgb"]["orientations"][:, np.newaxis]
                sensor_object = CoordinateSystem(rb_pos = rb_position, rb_ori = rb_orientation, length=0.05, color=(0.3, 0.3, 0.3, 1), icon="\u0086", name=sensor)
                renderables[sensor] = sensor_object
            else: 
                # sensor is RGB Camera
                targets = self.sensor_data["camera-rgb"]["positions"] + self.sensor_data["camera-rgb"]["orientations"] @ np.array([0,0,1])
                if self.try_open_cv_camera:
                    # OpenCVCamera - doesn't work yet
                    K = np.array(self.sensor_data["camera-rgb"]["K"])
                    Rt = np.array(self.sensor_data["camera-rgb"]["Rt"])
                    self.camera_rgb = OpenCVCamera(K=K, Rt=Rt, cols=viewer.window_size[0], rows=viewer.window_size[1], viewer=viewer)
                else:
                    # Pinhole Camera, projection does not yet overlap perfectly with billboard
                    self.camera_rgb = PinholeCamera(self.sensor_data["camera-rgb"]["positions"], targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer, fov=145, name=sensor)
                
                # For Pinhole Camera, update projection matrix afterwards to include the intrinsics for a perfect overlay with the billboard
                    
                renderables[sensor] = self.camera_rgb
        return renderables
    
    
    def get_rgb_billboard_renderable(self):
        assert hasattr(self, "rgb_camera"), "Call get_sensor_renderables() before get_rgb_billboard() and make sure the rgb camera exists."

        # prepare undistortion of fisheye rgb images for rendering
        camera_rgb_local_calibration = self.vrs_provider.get_device_calibration().get_camera_calib("camera-rgb")
        # Get projection matrix from fist calibration (shouldn't change much over time, therefore use first online calibration)
        original_image_size = self.sensor_data["camera-rgb"]["image_size"][0]
        focal_length = 400 # self.sensor_data["camera-rgb"]["K"][0][0][0]
        pinhole_calibration_for_undistortion = calibration.get_linear_camera_calibration(original_image_size, original_image_size, focal_length)
        if not self.try_open_cv_camera: # pinhole camera
            # it is important, that the target projection matrix for undistorting images shares the same focal_length as the pinhole camera used for rendering
            c1, c2 = pinhole_calibration_for_undistortion.get_principal_point() # camera center
            self.camera_rgb.update_matrices_known_intrinsics(width=original_image_size, height=original_image_size, f_1=focal_length, f_2=focal_length, c_1=c1, c_2=c2) # switch dimensions, as images will be rotated by 90 degrees before rendering
        else:
            self.camera_rgb.update_matrices(width=original_image_size, height=original_image_size)
        # define function to undistort and rotate 90 degrees during rendering
        def process_image(raw_image, _):
            undistorted_image = calibration.distort_by_calibration(raw_image, pinhole_calibration_for_undistortion, camera_rgb_local_calibration)
            processed_image = np.rot90(undistorted_image, k=3)
            return processed_image
        # Load each frame during rendering via vrs data provider
        # billboard = AriaBillboard.from_camera_and_distance(vrs_provider, timestamps_ns, camera_rgb, 1.34, original_image_size[1], original_image_size[0], np.zeros(len(targets)-1),
        #                                            image_process_fn = process_image)
        billboard = AriaBillboard.from_camera_and_distance(self.vrs_provider, self.timestamps_ns, self.camera_rgb, 1.34, original_image_size, original_image_size, np.zeros(len(self.sensor_data["camera-rgb"]["positions"])),
                                                    image_process_fn = process_image)
        billboard.texture_alpha = 0.9
        return billboard


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-C', '-closed_trajectory', type=bool, required=False, default=True, help="True if closed-loop trajectory should be visualized. If False, open-loop trajectory is used.")
    parser.add_argument('-T', '-trajectory_folder_path', type=str, required=False, default="/Users/annel/Documents/Github Repositories/aria_ait/data/1min_Trajectory_Trajectory", help="MPS folder name inside data_source path. Make sure to move the vrs Files into the folder")
    parser.add_argument('-F', '-frame_rate', type=int, required=False, default=60, help="Frame Rate to sample all data")
    parser.add_argument('-V', '-vrs_file-path', type=str, required=False, default=None, help="VRS file path. If not given, the script will look for a vrs file in the MPS folder.")
    parser.add_argument('-O', '-try_open_cv_camera', type=bool, required=False, default=False, help="Test case for opencv camera")
    args, _ = parser.parse_known_args()
    args = dict(map(lambda arg: (arg, getattr(args, arg)), vars(args)))
    
    aria_mesh = ProjectAriaSensors(
        trajectory_folder_path=args["T"],
        vrs_file_path=args["V"],
        closed_trajectory=args["C"],
        frame_rate=args["F"],
        try_open_cv_camera=args["O"]
    )
    
    C.update_conf({"playback_fps": args["F"]})
    C.update_conf({"scene_fps": args["F"]})
    
    v = Viewer()
    point_cloud = aria_mesh.get_point_cloud_renderable()
    sensor_renderables_dic = aria_mesh.get_sensor_renderables(v)
    billboard = aria_mesh.get_rgb_billboard_renderable()
    
    v.scene.add(point_cloud)
    for renderable in sensor_renderables_dic.keys():
        v.scene.add(sensor_renderables_dic[renderable])
    v.scene.add(billboard)
    v.run()