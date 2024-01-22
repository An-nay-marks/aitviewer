# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np
import argparse
import os
import datetime
import projectaria_tools.core.mps as mps

from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.configuration import CONFIG as C
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import bisection_timestamp_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-C', '-closed_trajectory', type=bool, required=False, default=True, help="True if closed-loop trajectory should be visualized. If False, open-loop trajectory is used.")
    parser.add_argument('-D', '-data_source', type=str, required=False, default="/Users/annel/Documents/Github Repositories/aria_ait/data", help="Path to where the all MPS folders are stored")
    parser.add_argument('-M', '-mps_folder', type=str, required=False, default="Profile2durationSlam_Trajectory", help="MPS folder name inside data_source path. Make sure to move the vrs Files into the folder")
    # TODO: add option to choose where vrs data is stored
    args, _ = parser.parse_known_args()
    args = dict(map(lambda arg: (arg, getattr(args, arg)), vars(args)))
    
    # Path convention
    vrs_file_name = args["M"][0:-11] + ".vrs"
    folder_name = os.path.join(args["D"], args["M"])
    
    vrs_file_path = os.path.join(folder_name, vrs_file_name)
    open_loop_trajectory_path = os.path.join(folder_name, "open_loop_trajectory.csv")
    closed_loop_trajectory_path = os.path.join(folder_name, "closed_loop_trajectory.csv")
    calibration_path = os.path.join(folder_name, "online_calibration.jsonl")
    semidense_pointcloud_path = os.path.join(folder_name, "semidense_points.csv.gz")
    semidense_observation_path = os.path.join(folder_name, "semidense_observations.csv.gz")
    
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
    timestamps_ns = np.linspace(start_time_device_ns, end_time_device_ns, int((end_time_device_ns - start_time_device_ns)*1e-6))
    
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
        pose = mps.utils.get_nearest_pose(trajectory, time_ns) # ClosedLoopTrajectoryPose or OpenLoopTrajectoryPose object that contains world coordinate frame graphUid
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
                try:
                    for senor in sensor_data.keys():
                        sensor_data[senor]["positions"].append(sensor_data[senor]["positions"][-1])
                        sensor_data[senor]["orientations"].append(sensor_data[senor]["orientations"][-1])
                except IndexError:
                    print("WARNING: No sensor calibration found for the first timestamp. Defaulting to (0,0,0) for all sensors.")
            else:
                for cam_calib in online_calibration.camera_calibs:
                    # its a camera -> functions between camera and other sensors are different
                    sensor_name = cam_calib.get_label()
                    T_device_sensor = cam_calib.get_transform_device_camera()
                    T_world_sensor = T_world_device @ T_device_sensor
                    SE3d_sensor_matrix = T_world_sensor.to_matrix()
                    
                    sensor_position_x_y_z = np.array([SE3d_sensor_matrix[1][3], SE3d_sensor_matrix[2][3], SE3d_sensor_matrix[0][3]])
                    sensor_data[sensor_name]["positions"][relative_timestamp] = sensor_position_x_y_z
                    
                    sensor_orientation = np.array([SE3d_sensor_matrix[1][0:3], SE3d_sensor_matrix[2][0:3], SE3d_sensor_matrix[0][0:3]])
                    sensor_data[sensor_name]["orientations"][relative_timestamp] = sensor_orientation
                
                for imu_sensor in online_calibration.imu_calibs:
                    # its an imu
                    sensor_name = imu_sensor.get_label()
                    T_device_sensor = imu_sensor.get_transform_device_imu()
                    T_world_sensor = T_world_device @ T_device_sensor
                    SE3d_sensor_matrix = T_world_sensor.to_matrix()
                    
                    sensor_position_x_y_z = np.array([SE3d_sensor_matrix[1][3], SE3d_sensor_matrix[2][3], SE3d_sensor_matrix[0][3]])
                    sensor_data[sensor_name]["positions"][relative_timestamp] = sensor_position_x_y_z
                    
                    sensor_orientation = np.array([SE3d_sensor_matrix[1][0:3], SE3d_sensor_matrix[2][0:3], SE3d_sensor_matrix[0][0:3]])
                    sensor_data[sensor_name]["orientations"][relative_timestamp] = sensor_orientation
        else:
            print("WARNING: No camera pose found for timestamp {}. This can lead to weird rendering artefacts, as the default pose is (0,0,0).".format(time_ns))
            
    
    # Get RGB image to project into scene (if available)
    if visualize_images and "camera-rgb" in sensor_data.keys() and os.path.exists(vrs_file_path):
        vrs_provider = data_provider.create_vrs_data_provider(vrs_file_path)
    else:
        print("No RGB image available. Check if sensor data available and vrs file exists in the data_source_path folder.")
        # TODO: get rgb image
    

    # Display in viewer, add framerate to 1000 fps, as we sample all data in ms-steps
    C.update_conf({"playback_fps": 1000})
    C.update_conf({"scene_fps": 1000})
    v = Viewer()
    
    #glasses = PinholeCamera(device_positions_x_y_z, targets, v.window_size[0], v.window_size[1], viewer=v) # TODO: create Glasses renderable class
    #v.scene.add(glasses)
    
    for sensor in sensor_data.keys():
        if sensor == "camera-rgb":
            # Only the rgb camera is visualized as a camera object with image projections
            targets = sensor_data["camera-rgb"]["positions"] + sensor_data["camera-rgb"]["orientations"] @ np.array([0,0,1])
            camera_rgb = PinholeCamera(sensor_data["camera-rgb"]["positions"], targets, v.window_size[0], v.window_size[1], viewer=v, name=sensor)
            # TODO: project rgb images into plane in front of camera
            v.scene.add(camera_rgb)

        if sensor != "camera-rgb":
            # visualize all other sensors as coordinate systems
            rb_position = np.expand_dims(sensor_data[sensor]["positions"], 1)
            # rb_orientation = rb_ori = np.repeat(np.eye(3)[np.newaxis, :], len(timestamps_ns), axis=0)[:, np.newaxis]
            rb_orientation = sensor_data["camera-rgb"]["orientations"][:, np.newaxis]
            sensor_object = CoordinateSystem(rb_pos = rb_position, rb_ori = rb_orientation, length=0.05, color=(1, 1, 1, 1), icon="\u0086", name=sensor)
            v.scene.add(sensor_object)
    
    # Set the camera as the current viewer camera.
    # v.set_temp_camera(glasses)

    v.run()
