# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np
import argparse
import os
import projectaria_tools.core.mps as mps

from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.billboard import AriaBillboard
from aitviewer.configuration import CONFIG as C
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import (
    bisection_timestamp_search,
    filter_points_from_confidence,
    get_nearest_pose
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-C', '-closed_trajectory', type=bool, required=False, default=True, help="True if closed-loop trajectory should be visualized. If False, open-loop trajectory is used.")
    parser.add_argument('-T', '-trajectory_folder_path', type=str, required=False, default="/Users/annel/Documents/Github Repositories/aria_ait/data/Profile2durationSlam_Trajectory", help="MPS folder name inside data_source path. Make sure to move the vrs Files into the folder")
    parser.add_argument('-F', '-frame_rate', type=int, required=False, default=60, help="Frame Rate to sample all data")
    parser.add_argument('-V', '-vrs_file-path', type=str, required=False, help="VRS file path. If not given, the script will look for a vrs file in the MPS folder.")
    args, _ = parser.parse_known_args()
    args = dict(map(lambda arg: (arg, getattr(args, arg)), vars(args)))
    
    # Path convention
    if args["V"] is not None:
        vrs_file_path = args["V"]
    else:
        vrs_files = []
        vrs_files += [each for each in os.listdir(args["T"]) if each.endswith('.vrs')]
        if len(vrs_files) == 0:
            raise FileNotFoundError("No vrs file found in the MPS folder. Please specify the path to the vrs file with the -V argument.")
        vrs_file_path = os.path.join(args["T"], vrs_files[0])
    open_loop_trajectory_path = os.path.join(args["T"], "open_loop_trajectory.csv")
    closed_loop_trajectory_path = os.path.join(args["T"], "closed_loop_trajectory.csv")
    calibration_path = os.path.join(args["T"], "online_calibration.jsonl")
    semidense_pointcloud_path = os.path.join(args["T"], "semidense_points.csv.gz")
    semidense_observation_path = os.path.join(args["T"], "semidense_observations.csv.gz")
    
    visualize_images = True
    visualize_pointcloud = True
    
    # Check paths
    for p in [open_loop_trajectory_path, closed_loop_trajectory_path, calibration_path]:
        assert os.path.exists(p), "Path {} does not exist.".format(p)
    
    if not os.path.exists(vrs_file_path):
        print("Did not find vrs file. Won't be able to visualize the pictures and the exact sensors.")
        visualize_images = False
    
    if not os.path.exists(semidense_pointcloud_path):
        print("Did not find semi-dense pointcloud data. Won't be able to visualize the pointcloud.")
        visualize_pointcloud = False
     
    # Load open or closed loop trajectory
    if args["C"]:
        closed_loop_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory_path)
        trajectory = closed_loop_trajectory
    else:
        open_loop_trajectory = mps.read_open_loop_trajectory(open_loop_trajectory_path)
        trajectory  = open_loop_trajectory
    
    # create timestamps for each millisecond (but unit is nanoseconds)s
    start_time_device_ns = int((trajectory[0].tracking_timestamp.total_seconds() + 1) * 1e9) # add 1 ns to avoid having no pose for the first timestamp
    end_time_device_ns = int(trajectory[-1].tracking_timestamp.total_seconds() * 1e9)
    fps_to_nspf = 1e9 / args["F"] # 1e9 nanoseconds per second, 1 / fps frames per second
    timestamps_ns = np.linspace(start_time_device_ns, end_time_device_ns, int((end_time_device_ns - start_time_device_ns)/fps_to_nspf))
    
    # Get the Online Calibration for each timestamps
    online_calibrations_filtered = []
    online_calibs = mps.read_online_calibration(calibration_path)
    for index, time_ns in enumerate(timestamps_ns): # Online Calibration returns a list of device calibrations
        nearest_index = bisection_timestamp_search(online_calibs, time_ns)
        if nearest_index is None:
            online_calibrations_filtered.append(None)
        else:
            online_calibrations_filtered.append(online_calibs[nearest_index])
    
    # Create dictionary of all existing sensors (checked via Online Calibration) to save their positions and orientations
    sensor_data = {}
    for cal in online_calibrations_filtered:
        if cal is not None:
            sample_calibration = cal
            break
    for cam in sample_calibration.camera_calibs:
        sensor_data[cam.get_label()] = {"positions": np.zeros((len(timestamps_ns), 3)),
                                        "orientations": np.repeat(np.eye(3)[np.newaxis, :], len(timestamps_ns), axis=0)}
    for imu in sample_calibration.imu_calibs:
        sensor_data[imu.get_label()] = {"positions": np.zeros((len(timestamps_ns), 3)),
                                        "orientations": np.repeat(np.eye(3)[np.newaxis, :], len(timestamps_ns), axis=0)}
        
    
    # Get positions and orientation of the device and all sensors
    device_positions_x_y_z = np.zeros((len(timestamps_ns), 3))
    device_orientation_x_y_z = np.zeros((len(timestamps_ns), 3, 3))
    # A trajectory is just a list of poses
    for relative_timestamp, time_ns in enumerate(timestamps_ns): # relative_timestamp is the index of the timestamp, describing the time since record-start, rather than the actual timestamp of the device
        pose = get_nearest_pose(trajectory, time_ns) # ClosedLoopTrajectoryPose or OpenLoopTrajectoryPose object that contains world coordinate frame graphUid
        if pose:
            T_world_device = pose.transform_world_device # Sophus::SE3d object
            SE3d_matrix = T_world_device.to_matrix() # 
            device_position_x_y_z = np.array([SE3d_matrix[1][3], SE3d_matrix[2][3], SE3d_matrix[0][3]])
            device_positions_x_y_z[relative_timestamp] = device_position_x_y_z
            device_orientation_x_y_z[relative_timestamp] = np.array([SE3d_matrix[1][0:3], SE3d_matrix[2][0:3], SE3d_matrix[0][0:3]])
            # Get each sensor calibration from online calibrations
            online_calibration = online_calibrations_filtered[relative_timestamp]
            if online_calibration is None:
                print(f"WARNING: No online sensor calibration found for timestamp {time_ns}ns. Relative Calibration to the device from the previous timestamp will be copied.")
                for senor in sensor_data.keys():
                    sensor_data[senor]["positions"][relative_timestamp] = (sensor_data[senor]["positions"][relative_timestamp-1])
                    sensor_data[senor]["orientations"][relative_timestamp] = (sensor_data[senor]["orientations"][relative_timestamp-1])
            else:
                for cam_calib in online_calibration.camera_calibs:
                    # its a camera -> functions between camera and other sensors are different
                    sensor_name = cam_calib.get_label()
                    T_device_sensor = cam_calib.get_transform_device_camera()
                    T_world_sensor = T_world_device @ T_device_sensor
                    T_world_sensor_matrix = T_world_sensor.to_matrix()
                    
                    sensor_position_x_y_z = np.array([T_world_sensor_matrix[1][3], T_world_sensor_matrix[2][3], T_world_sensor_matrix[0][3]])
                    sensor_data[sensor_name]["positions"][relative_timestamp] = sensor_position_x_y_z
                    
                    sensor_orientation = np.array([T_world_sensor_matrix[1][0:3], T_world_sensor_matrix[2][0:3], T_world_sensor_matrix[0][0:3]])
                    sensor_data[sensor_name]["orientations"][relative_timestamp] = sensor_orientation
                    
                    # Save full calibration for projection rgb images into the scene
                    if sensor_name == "camera-rgb":
                        if "full_calibration" not in sensor_data[sensor_name].keys():
                            sensor_data[sensor_name]["full_calibration"] = []

                        sensor_data[sensor_name]["full_calibration"].append(cam_calib)
                
                for imu_sensor in online_calibration.imu_calibs:
                    # its an imu
                    sensor_name = imu_sensor.get_label()
                    T_device_sensor = imu_sensor.get_transform_device_imu()
                    T_world_sensor = T_world_device @ T_device_sensor
                    T_world_sensor_matrix = T_world_sensor.to_matrix()
                    
                    sensor_position_x_y_z = np.array([T_world_sensor_matrix[1][3], T_world_sensor_matrix[2][3], T_world_sensor_matrix[0][3]])
                    sensor_data[sensor_name]["positions"][relative_timestamp] = sensor_position_x_y_z
                    
                    sensor_orientation = np.array([T_world_sensor_matrix[1][0:3], T_world_sensor_matrix[2][0:3], T_world_sensor_matrix[0][0:3]])
                    sensor_data[sensor_name]["orientations"][relative_timestamp] = sensor_orientation
        else:
            print("WARNING: No camera pose found for timestamp {}. This can lead to weird rendering artefacts, as the default pose is (0,0,0).".format(time_ns))

    # Display in viewer, add framerate, as we sample all data in ms-steps depending on user input
    C.update_conf({"playback_fps": args["F"]})
    C.update_conf({"scene_fps": args["F"]})
    v = Viewer()
    
    # Add point cloud
    if visualize_pointcloud:
        points = mps.read_global_point_cloud(semidense_pointcloud_path)
        points = filter_points_from_confidence(points) # Filter by inv depth and depth to only show points, where the algorithm is confident enough about its 3d position
        point_cloud_np = np.stack([np.array([x.position_world[1],x.position_world[2],x.position_world[0]]) for x in points])[np.newaxis, :] # shape (F, N, 3) with y, z, x
        point_cloud = PointClouds(point_cloud_np, color=(0, 0, 0, 0.8), name="point_cloud")
        v.scene.add(point_cloud)
    
    # TODO: create Glasses renderable class
    
    # Add all available sensors to the scene. For RGB Camera, add a camera model and the images additionally
    for sensor in sensor_data.keys():
        if sensor != "camera-rgb":
            # visualize all other sensors as coordinate systems
            rb_position = np.expand_dims(sensor_data[sensor]["positions"], 1)
            # rb_orientation = rb_ori = np.repeat(np.eye(3)[np.newaxis, :], len(timestamps_ns), axis=0)[:, np.newaxis]
            rb_orientation = sensor_data["camera-rgb"]["orientations"][:, np.newaxis]
            sensor_object = CoordinateSystem(rb_pos = rb_position, rb_ori = rb_orientation, length=0.05, color=(0.3, 0.3, 0.3, 1), icon="\u0086", name=sensor)
            v.scene.add(sensor_object)
        else: # RGB Camera
            
            # Visualized as a camera object with image projections
            targets = sensor_data["camera-rgb"]["positions"] + sensor_data["camera-rgb"]["orientations"] @ np.array([0,0,1])
            camera_rgb = PinholeCamera(sensor_data["camera-rgb"]["positions"], targets, v.window_size[0], v.window_size[1], viewer=v, fov=145, name=sensor)
            v.scene.add(camera_rgb)
            
            # Project images into Pinhole Camera
            if visualize_images:
                
                vrs_provider = data_provider.create_vrs_data_provider(vrs_filename = str(vrs_file_path))
                
                rgb_stream_id = vrs_provider.get_stream_id_from_label("camera-rgb")
                original_image_size = sensor_data["camera-rgb"]["full_calibration"][0].get_image_size()
                camera_rgb.update_matrices(width=original_image_size[1], height=original_image_size[0]) # switch dimensions, as images will be rotated by 90 degrees before rendering
                
                # define function to undistort and rotate 90 degrees during rendering
                def process_image(raw_image, frame_idx):
                    online_calibration = sensor_data["camera-rgb"]["full_calibration"][frame_idx]
                    focal_length = camera_rgb.get_projection_matrix()[0,0]
                    pinhole_mps = calibration.get_linear_camera_calibration(raw_image.shape[0], raw_image.shape[0], focal_length)
                    undistorted_image = calibration.distort_by_calibration(raw_image, pinhole_mps, online_calibration)
                    processed_image = np.rot90(undistorted_image, k=3)
                    return processed_image
                
                # Load each frame during rendering via vrs data provider
                billboard = AriaBillboard.from_camera_and_distance(vrs_provider, timestamps_ns, camera_rgb, 1.34, original_image_size[1], original_image_size[0], np.zeros(len(targets)),
                                                            image_process_fn = process_image)
                billboard.texture_alpha = 0.9
                v.scene.add(billboard)

    v.run()
