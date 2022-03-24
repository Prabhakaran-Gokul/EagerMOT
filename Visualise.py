from numpy.core.defchararray import center
from numpy.core.fromnumeric import take
from numpy.lib.shape_base import apply_along_axis
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes

# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering

import numpy as np
from typing import List, Mapping
from pyquaternion import Quaternion
from PIL import Image, ImageFont, ImageDraw
import os
import json
from numba import njit
import cv2
from matplotlib import pyplot as plt
import time
import math
import vedo

# nusc = NuScenes(version='v1.0-mini', dataroot='C:\\Users\\prabh\\Documents\\Y4Sem1\\FYP\\mini', verbose=True)
# nusc.list_scenes()
# scenes = nusc.scene
# my_scene = scenes[-4]

# last_sample_token = my_scene['last_sample_token'] 
# my_sample = nusc.get('sample', last_sample_token)
# frame_data = my_sample["data"]
# lidar_top_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
# lidar_sensor_data = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])

# pcd_file = "C:\\Users\\prabh\\Documents\\Y4Sem1\\FYP\\mini\\samples\\LIDAR_TOP\\n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984253447765.pcd.bin"

class Visualise():
    def __init__(self) -> None:
        self.dataset_path = 'C:\\Users\\prabh\\Documents\\Y4Sem1\\FYP\\mini'
        self.nusc = NuScenes(version='v1.0-mini', dataroot=self.dataset_path, verbose=True)
        self.tracking_results_file = "v1.0-mini_tracking.json"
        self.tracking_results = None 
        self.vi = vedo.Plotter(bg = (255, 255, 255), offscreen=True) #change offscreen to True if saving video using vedo

    def get_lidar_pcd_files(self):
        pass

    def get_all_scenes(self):
        all_scenes = {}
        for scene in self.nusc.scene:
            all_scenes[scene["name"]] = scene
        return all_scenes 
    
    def get_all_frame_tokens(self, scene):
        frame_tokens = []
        frame_token = scene['first_sample_token']
        while frame_token:
            frame_nu = self.nusc.get("sample", frame_token)
            frame_tokens.append(frame_token)
            assert frame_nu["scene_token"] == scene["token"]
            frame_token = frame_nu["next"]

        assert len(frame_tokens) == scene["nbr_samples"]
        return frame_tokens 

    def world_from_lidar(self, lidar_points: np.ndarray, frame_data: Mapping) -> np.ndarray:
        """
        :param lidar_points: Nx3 points as np.ndarray
        :return: [world 3D points centered around origin] and [original mean point in world frame]
        """
        # from lidar frame to ego frame
        lidar_data = self.nusc.get('sample_data', frame_data["LIDAR_TOP"])
        ego_points = self.lidar_to_ego(lidar_points, lidar_data)

        # from ego frame to world frame
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        # print ("ego_pose_data")
        # print(ego_pose_data)
        return self.transform_points_with_pose(ego_points, ego_pose_data)

    def lidar_to_ego(self, lidar_points: np.ndarray, lidar_data: Mapping) -> np.ndarray:
        lidar_sensor_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        return self.transform_points_with_pose(lidar_points, lidar_sensor_data)

    def transform_points_with_pose(self, points: np.ndarray, pose_data: Mapping) -> np.ndarray:
        result = points @ Quaternion(pose_data['rotation']).rotation_matrix.T
        result += np.array(pose_data['translation'])
        return result

    def load_tracking_results(self):
        file_path = os.path.join("C:\\Users\\prabh\Documents\\Y4Sem1\\FYP", self.tracking_results_file)
        print (f"Loading {file_path}")
        with open(file_path, 'r') as f:
            full_results_json = json.load(f)
        tracking_results = full_results_json["results"]
        self.tracking_results = tracking_results
        print(f"Done loading {file_path}")

    def get_tracks(self, frame):
        if self.tracking_results == None:
            self.load_tracking_results()
        assert self.tracking_results != None 
        return self.tracking_results[frame]

    def get_sensor_data_file(self, frame_token, sensor: str):
        my_sample = self.nusc.get('sample', frame_token)
        sensor_data = self.nusc.get('sample_data', my_sample['data'][sensor])
        sensor_file = sensor_data["filename"]
        return sensor_file

    def get_frame_data(self, sample_token):
        my_sample = self.nusc.get('sample', sample_token)
        return my_sample["data"]

    def create_vid(self, imgs, scene_name, cam):
        if not imgs: 
            return "No images to create video from."
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        vw = cv2.VideoWriter(f"{scene_name}_{cam}.avi", fourcc, 2, (1600, 900))
        for img in imgs:
            vw.write(img)
            cv2.imshow("show", img)
            cv2.waitKey(0)
        vw.release()

    def show_2d_vid(self, scene_name, cam: str):
        scene = self.get_all_scenes()[scene_name]
        frame_tokens = self.get_all_frame_tokens(scene)
        imgs = []
        for frame_token in frame_tokens:
            img = self.show_frame_2d(frame_token, cam)
            imgs.append(img)
        
        self.create_vid(imgs, scene_name, cam)

    def show_frame_2d(self, frame, cam: str):
        #get corners of bboxes and convert them to pixel coordinates
        all_corners, track_names, tracking_ids = self.get_all_bbox_corners(frame)
        frame_data = self.get_frame_data(frame)
        pixel_pts, names_filtered, ids_filtered = self.img_from_tracking(all_corners, cam, frame_data, track_names, tracking_ids)
        
        #filter pixel coordinates out of the image
        cam_data = self.nusc.get('sample_data', frame_data[cam]) 
        filtered_pts, names_filtered, ids_filtered = self.filter_pixels(pixel_pts, cam_data['width'], cam_data['height'], names_filtered, ids_filtered) 

        #get image 
        img_path = self.get_sensor_data_file(frame, sensor=cam)
        img_path = os.path.join(self.dataset_path, img_path)
        image = cv2.imread(img_path)

        img_with_bboxes = self.draw_all_bbox(image, filtered_pts, names_filtered, ids_filtered)
        return img_with_bboxes

    # def show_frame_3d(self, frame):
    #     tracks = self.get_tracks(frame)
    #     frame_data = self.nusc.get('sample', frame)["data"]
    #     bboxes = []

    #     for track in tracks:
    #         center = track["translation"]
    #         size = track["size"]
    #         orientation = Quaternion(track["rotation"])
    #         score = track["tracking_score"]
    #         name = track["tracking_name"]
    #         velocity = track["velocity"] 

    #         bbox_corners = self.get_bbox_corners(center, size, orientation, score, velocity, name)
    #         bboxes.append(get_line_set_o3d(bbox_corners))
    #         # bboxes.append(get_bbox_vedo(bbox_corners))
    #     # center_pcd = text_3d("Hello", center, font_size=500, density=1) 
    #     pcd_file = os.path.join(self.dataset_path, self.get_sensor_data_file(frame_token=frame, sensor='LIDAR_TOP'))
    #     all_pcd_points = np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]

    #     transformed_pts = self.world_from_lidar(all_pcd_points, frame_data)
    #     o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_pts))
    #     o3d.visualization.draw_geometries([*bboxes, o3d_pcd])

    #     # actors = []
    #     # actors.append(vedo.Points(transformed_pts, r = 1))
    #     # actors.extend(bboxes)
    #     # self.vi.show(actors)

    def show_3d_vid(self, scene_name):
        scene = self.get_all_scenes()[scene_name]
        frame_tokens = self.get_all_frame_tokens(scene)
        video = vedo.Video(f"3d_{scene_name}.mp4", duration=len(frame_tokens)/2, backend='opencv')
        print("Creating video...")
        angle = math.radians(0)
        focal_pt_vec = (0, math.cos(angle), math.sin(angle))

        for frame_token in frame_tokens:
            actors = self.show_frame_3d(frame_token)
            # self.vi.show(actors, camera={'pos': (0, 0, 0), 'focalPoint': focal_pt_vec, 'viewup': (0, 0, 1)})
            self.vi.show(actors, camera={'pos': (0, -5, 0), 'viewup': (0, 0, 1)})
            video.addFrame()
            
        video.close()

    def show_frame_3d(self, frame):
        all_corners, track_names, track_ids = self.get_all_bbox_corners(frame)

        # Move the pcd(corners of bboxes) from world to ego frame
        lidar_data = self.nusc.get('sample_data', self.get_frame_data(frame)["LIDAR_TOP"])
        ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_points = inverse_transform_points_with_pose(all_corners, ego_pose_data)

        # Move the pcd(corners of bboxes) from ego to lidar sensor frame
        lidar_sensor_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        lidar_corner_pts = inverse_transform_points_with_pose(ego_points, lidar_sensor_data)

        lidar_corner_pts = np.reshape(lidar_corner_pts, (-1, 8, 3))
        
        #TODO Check the possibilty of using np.apply_along_axis function to map
        bboxes_vedo = []
        txt_vedo = []
        for idx, corner in enumerate(lidar_corner_pts):
            lines = self.get_bbox_vedo(corner, track_names[idx])
            bboxes_vedo.append(lines)
            txt_vedo.append(self.get_txt_vedo(corner, track_names[idx], track_ids[idx]))
        # bboxes_vedo = [get_bbox_vedo(corner) for corner in lidar_corner_pts]
        actors = []

        #add bboxes and txt to actors
        actors.extend(bboxes_vedo)
        actors.extend(txt_vedo)

        #add all pcd data to actors 
        pcd_file = os.path.join(self.dataset_path, self.get_sensor_data_file(frame_token=frame, sensor='LIDAR_TOP'))
        all_pcd_points = np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]
        actors.append(vedo.Points(all_pcd_points, r = 3))

        return actors

    def get_all_bbox_corners(self, frame):
        tracks = self.get_tracks(frame)
        all_corners = []
        bbox_names = []
        tracking_ids = []
        for track in tracks:
            center = track["translation"]
            size = track["size"]
            orientation = Quaternion(track["rotation"])
            score = track["tracking_score"]
            name = track["tracking_name"]
            velocity = track["velocity"] 
            tracking_id = track["tracking_id"]

            bbox_corners = self.get_bbox_corners(center, size, orientation, score, velocity, name)
            all_corners.extend(bbox_corners)
            bbox_names.append(name)
            tracking_ids.append(tracking_id)
        return np.array(all_corners), np.array(bbox_names), np.array(tracking_ids)

    #get the corners of a single bbox
    def get_bbox_corners(self, center, size, orientation, score, velocity, name):
        box = Box(center, size, orientation, score=score, velocity=velocity, name=name)
        corners = box.corners()
        corners = np.transpose(corners)
        return corners

    def img_from_tracking(self, track_points: np.ndarray, cam: str, frame_data, track_names: np.ndarray, tracking_ids: np.ndarray) -> np.ndarray:
        """
        :param track_points: nx3 3D points in the tracking frame i.e. world coordinates in KITTI rect frame 
        :param camera: to which camera plane perform the projection
        :return: nx2 2D coordinates of points in the specified camera's image coordinates
        """
        assert frame_data is not None

        # Rotate points from KITTI frame to NuScenes frame, coordinates are relative to world origin
        # nuscenes_world_points = kitti_to_nuscenes(track_points)

        # Move the pcd from world to ego frame
        cam_data = self.nusc.get('sample_data', frame_data[cam])
        ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_points = inverse_transform_points_with_pose(track_points, ego_pose_data)

        # Move the pcd from ego to cam sensor frame
        # cam_data = self.nusc.get('sample_data', frame_data[cam])
        cam_sensor_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_points = inverse_transform_points_with_pose(ego_points, cam_sensor_data)

        # Only keep points in front of the frame
        # cam_front_points = cam_points[cam_points[:, 2] > 0]
        cam_front_points, names_filtered, ids_filtered = self.filter_points(cam_points, track_names, tracking_ids)

        # Project to image plane
        intrinsic = np.array(cam_sensor_data['camera_intrinsic'])
        assert intrinsic.shape == (3, 3)
        img_points = cam_front_points @ intrinsic.T

        return cam_points_to_image_coordinates(img_points).astype(int, copy=False), names_filtered, ids_filtered

    def filter_points(self, points, track_names: np.ndarray, tracking_ids: np.ndarray):
        """
        :param points nx3 3D points in camera frame
        """
        #convert points to (n, 8, 3) shape where n = total number of points/8
        points_reshaped = np.reshape(points, (-1, 8, 3))
        mask = (points_reshaped[:, :, 2] > 0).all(axis = 1) 
        #Only keep points in front of the camera frame
        pts_in_front = points_reshaped[mask]
        names_filtered = track_names[mask]
        ids_filtered = tracking_ids[mask]

        #filter the points and return array with original nx3 shape
        result = np.reshape(pts_in_front, (-1, 3))
        return result, names_filtered, ids_filtered

    def filter_pixels(self, pixel_coordinates, width_limit, height_limit, track_names, track_ids):
        pts = np.reshape(pixel_coordinates, (-1, 8, 2))
        #remove pts with width out of limit
        mask_for_width = ((pts[:, :, 0] >= 0) & (pts[:, :, 0] <= width_limit)).all(axis=1)
        pts = pts[mask_for_width]
        track_names = track_names[mask_for_width]
        track_ids = track_ids[mask_for_width]

        #remove pts with height out of limit
        mask_for_height = ((pts[:, :, 1] >= 0) & (pts[:, :, 1] <= height_limit)).all(axis=1)
        pts = pts[mask_for_height]
        track_names = track_names[mask_for_height]
        track_ids = track_ids[mask_for_height]

        # result = np.reshape(pts, (-1, 2))
        return pts, track_names, track_ids

    def draw_3D_bbox_on_img(self, img, corners, color, name, id):
        assert len(corners) == 8
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [4, 0],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        #draw bbox
        for start_pt, end_pt in edges:
            cv2.line(img, corners[start_pt], corners[end_pt], color=color, thickness=1)
        
        #add text
        org = ((corners[:, 0].max() - corners[:, 0].min()) // 2 + corners[:, 0].min(), corners[:, 1].min() - 5)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.3
        text = f"{name} ({id})"
        cv2.putText(img, text, org, fontFace=font, color=color, thickness=1, lineType=4, fontScale=font_scale)

        return img

    def draw_all_bbox(self, img, all_corners, names_filtered, ids_filtered):
        assert len(all_corners) == len(names_filtered) == len(ids_filtered)
        for i in range(len(all_corners)):
            # print(names_filtered[i])
            color = self.get_bbox_color(names_filtered[i])
            img = self.draw_3D_bbox_on_img(img, all_corners[i], color, names_filtered[i], ids_filtered[i])
        return img
    
    def get_txt_vedo(self, corner_pts, track_name, track_id):
        cx, cy, cz = np.mean(corner_pts, axis=0)
        max_z = np.max(corner_pts[:, 2])
        z_padding = 0.1
        BGR = self.get_bbox_color(track_name) 
        assert len(BGR) == 3
        RGB = [BGR[2], BGR[1], BGR[0]]
        vedo_txt = vedo.Text3D(f"{track_name} ({str(track_id)})", s=0.15, c=RGB, justify="centered").rotateX(90).pos(cx, cy, max_z + z_padding)
        return vedo_txt

    def get_bbox_vedo(self, corner_pts, track_name):
        assert len(corner_pts) == 8
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [4, 0],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        lines_np = np.array(lines)
        start_pt_idx = lines_np[:, 0]
        end_pt_idx = lines_np[:, 1]

        start_pts = corner_pts[start_pt_idx]
        end_pts = corner_pts[end_pt_idx]
        bbox_BGR = self.get_bbox_color(track_name) 
        assert len(bbox_BGR) == 3
        bbox_RGB = [bbox_BGR[2], bbox_BGR[1], bbox_BGR[0]]
        lines = vedo.Lines(startPoints=start_pts, endPoints=end_pts, c=bbox_RGB)
        

        # lines.caption(txt=f"{track_name} ()", c=bbox_RGB, size=(0.1, 0.1), point=np.mean(corner_pts, axis=0), justify='left')
        # lines._caption.SetBorder(False)
        # lines._caption.SetLeader(False)
        return lines

    def get_bbox_color(self, category: str):
        """
        :param category: caterory of the track (e.g. car, bus etc.)
        :return: BGR tuple
        """
        result = (0, 0, 0) #return black color if the catergory in invalid
        for key in self.nusc.colormap:
            key_words = key.split(".")
            if category in key_words:
                R, G, B = self.nusc.colormap[key]
                result = (B, G, R)
                return result
        return result


@njit
def cam_points_to_image_coordinates(img_points):
    """
    :param img_points: nx3 3D points in camera frame coordinates
    :return: nx2 2D coordinates of points in image coordinates
    """
    img_points[:, 0] /= img_points[:, 2]
    img_points[:, 1] /= img_points[:, 2]
    # img_points = img_points[:, :2] / img_points[:, 2].reshape(-1, 1)
    img_plane_points = np.rint(img_points)
    return img_plane_points[:, :2]

def inverse_transform_points_with_pose(points: np.ndarray, pose_data) -> np.ndarray:
    result = points.copy()
    result -= np.array(pose_data['translation'])
    return result @ Quaternion(pose_data['rotation']).rotation_matrix


def combine_vids(twoD_vid, threeD_vid, scene_num):
    first_cap = cv2.VideoCapture(twoD_vid)
    second_cap = cv2.VideoCapture(threeD_vid)

    while ((not first_cap.isOpened()) and (not second_cap.isOpened())):
        first_cap = cv2.VideoCapture(twoD_vid)
        second_cap = cv2.VideoCapture(threeD_vid)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    vw = cv2.VideoWriter(f"combined_scene-{scene_num}.avi", fourcc, 2, (1600, 900 * 2))

    while (first_cap.isOpened() and first_cap.isOpened()):
        ret1, frame1 = first_cap.read()
        ret2, frame2 = second_cap.read()

        if (ret1 and ret2):
            h1, w1, c1 = frame1.shape
            h2, w2, c2 = frame2.shape

            if h1 != h2 or w1 != w2: # resize right img to left size
                frame2 = cv2.resize(frame2, (w1,h1))
        
            combined_frame = cv2.vconcat([frame1, frame2])  
            # cv2.imshow("ji", combined_frame)
            # cv2.waitKey(0)
            vw.write(combined_frame)
        else:
            break

    vw.release()




vs = Visualise()
# frame = "b4ff30109dd14c89b24789dc5713cf8c"
# vs.show_frame_3d(frame)
# vs.show_3d_vid(scene_name="scene-0916") 
# vs.show_frame_2d(frame, "CAM_BACK")
vs.show_2d_vid(scene_name="scene-0916", cam="CAM_FRONT")
# vs.show_2d_vid(scene_name="scene-0103", cam="CAM_BACK_RIGHT") 

# combine_vids("scene-0916_CAM_FRONT.avi", "3d_scene-0916.mp4", scene_num="0916")