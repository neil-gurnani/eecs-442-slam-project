import numpy as np
import cv2
import pdb
import math

from geometry import *

class MapPoint():
	def __init__(self, pos, descriptor):
		pos = np.array(pos).reshape(-1,1)
		if len(pos) == 3:
			pos = homogenize_vectors(pos)
		self.pos = pos # Should be a homogeneous 4x1 column vector
		self.descriptor = descriptor

	def __repr__(self):
		return "<MapPoint pos:%s descriptor:%s>" % (self.pos.flatten()[:-1], self.descriptor)

class Frame():
	def __init__(self, img, keypoints, descriptors, intrinsic_mat, t=None, index=None):
		self.img = img
		self.keypoints = keypoints # Should be a 3x1 set of homogeneous 2D vectors (in the image plane)
		                           # By convention, the center of the top-left corner pixel is (0,0)
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints
		self.intrinsic_mat = intrinsic_mat
		self.t = t
		self.index = index

class Map():
	def __init__(self):
		self.frames = []
		self.camera_poses = []
		self.map_points = []
		# Need some sort of data structure holding point correspondences

	def add_map_points(self, map_points):
		# Add a list of map_points
		self.map_points = self.map_points + map_points

	def add_frame(self, frame, pose):
		self.frames.append(frame)
		self.camera_poses.append(pose)

class SLAM():
	def __init__(self, match_descriptors_func):
		self.local_map = Map()
		self.global_map = Map()
		self.has_finished_initialization = False
		self.match_descriptors = match_descriptors_func # This function should take in two lists of descriptors,
		                                                # and return a nx2 numpy array of pairs of indices of matches.

	def start_initialization(self, frame, ground_truth_pose):
		# The ground truth camera pose is only used in the first frame, so that we can work in the same
		# coordinate system as the ground truth outputs
		self.init_frame = frame
		self.init_pose = ground_truth_pose

	def try_finish_initialization(self, frame, scale):
		# Get possible matches
		pairs = self.match_descriptors(self.init_frame.descriptors, frame.descriptors)
		start_points, next_points = self.init_frame.keypoints[:,pairs[:,0]], frame.keypoints[:,pairs[:,1]]
		start_points, next_points = start_points[:-1,:], next_points[:-1,:]
		descriptors = self.init_frame.descriptors[pairs[:,0]]

		# Normalize
		point_4d, R, t, mask = triangulate(start_points, next_points, frame.intrinsic_mat)
		t = t * scale # Scale for the first displacement only
		descriptors = descriptors[mask]

		# Compue the camera and point positions in the global frame
		mat = np.matmul(make_translation_matrix(t), homogenize_matrix(R)) # Maps from the old camera frame to the new camera frame
		new_pos = np.matmul(mat, self.init_pose.pos)
		new_quat = mat_to_quat(unhomogenize_matrix(np.matmul(mat, homogenize_matrix(quat_to_mat(self.init_pose.quat)))))
		new_pose = Pose(new_pos, new_quat, t=frame.t)
		points_global = local_xyz_to_global_xyz(new_pose, point_4d)
		map_points = [MapPoint(points_global[:,i], descriptors[i]) for i in range(len(descriptors))]

		# Compute the angle between the two frames for each point
		start_vecs = (points_global - self.init_pose.pos)[0:3]
		next_vecs = (points_global - new_pose.pos)[0:3]
		start_vecs = start_vecs / np.linalg.norm(start_vecs, axis=0)
		next_vecs = next_vecs / np.linalg.norm(next_vecs, axis=0)
		dprods = np.sum(np.multiply(start_vecs, next_vecs), axis=0)

		# import matplotlib.pyplot as plt
		# plt.scatter(start_points[0], start_points[1], color="blue", s=20**2)
		# plt.scatter(local_xyz_to_uv(frame.intrinsic_mat, point_4d)[0], local_xyz_to_uv(frame.intrinsic_mat, point_4d)[1], color="red")
		# plt.show()

		# Remove points with insufficient parallax
		cos1 = math.cos(1.0 * (math.pi / 180.0))
		cos2 = math.cos(2.0 * (math.pi / 180.0))
		mask = dprods < cos1

		if np.sum(mask) < 40: # Check that we have enough points
			print("Not enough parallax points! Only found %d." % np.sum(mask))
			return
		elif np.mean(dprods[mask]) > cos2: # Check that the points we have are good overall
			print("Average parallax insufficient! Needed %f, got %f." % (math.acos(cos2) * 180 / math.pi, math.acos(np.mean(dprods[mask])) * 180 / math.pi))
			return
		else:
			print("Found sufficient parallax! Finishing initialization.")
			self.has_finished_initialization = True
			for map_obj in [self.local_map, self.global_map]:
				map_obj.add_map_points(map_points)
				map_obj.add_frame(self.init_frame, self.init_pose)
				map_obj.add_frame(frame, new_pose)
			return

	def track_next_frame(self, frame):
		# Find map points visible in the current frame, by looking at the previous frame's pose.
		map_point_coords = np.array([point.pos.flatten() for point in self.local_map.map_points]).T
		local_coords = global_xyz_to_local_xyz(self.local_map.camera_poses[-1], map_point_coords)
		uv_points, idx = local_only_good_image_idx(frame.intrinsic_mat, frame.img.shape, local_coords)
		descriptors = np.array([point.descriptor for point in self.local_map.map_points])[idx]

		# Match frame keypoints with map points
		pairs = self.match_descriptors(frame.descriptors, descriptors)
		frame_points, map_points = frame.keypoints[:,pairs[:,0]], map_point_coords[:,pairs[:,1]]
		frame_points, map_points = frame_points[:-1,:], map_points[:-1,:]

		# Reshape according to this: https://stackoverflow.com/questions/33696082/error-using-solvepnpransac-function
		frame_points = frame_points.T.reshape(-1,1,2)
		map_points = map_points.T.reshape(-1,1,3)
		camera_mat_3x3 = frame.intrinsic_mat[0:3,0:3]

		# Actually find the camera position
		suc, R, t, mask = cv2.solvePnPRansac(map_points, frame_points, camera_mat_3x3, None)
		# R, t are the rotation and translation from the camera frame to the world frame
		# So in our pose scheme, they're literally the pose of the camera
		camera_pose = Pose(t, rot_vec_to_quat(R), t=frame.t)
		for map_obj in [self.local_map, self.global_map]:
			map_obj.add_frame(frame, camera_pose)