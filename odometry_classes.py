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
	def __init__(self, img, keypoints, descriptors, intrinsic_mat, t=None, index=None, use_opencv_keypoints=True):
		self.img = img
		self.keypoints = keypoints
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints
		self.intrinsic_mat = intrinsic_mat
		self.t = t
		self.index = index
		if use_opencv_keypoints:
			self.get_coords_from_keypoints()
		else:
			self.keypoint_coords = self.keypoints

	def get_coords_from_keypoints(self):
			# Should return a 3xn set of homogeneous 2D vectors (in the image plane)
			# By convention, the center of the top-left corner pixel is (0,0)
			self.keypoint_coords = np.ones((3,len(self.keypoints)))
			for i in range(len(self.keypoints)):
				self.keypoint_coords[0:2,i] = self.keypoints[i].pt

class Map():
	def __init__(self, max_map_points=np.inf):
		self.frames = []
		self.camera_poses = []
		self.map_points = []
		self.max_map_points = max_map_points
		self.last_keyframe_idx = None
		self.map_point_last_checked = []
		# Need some sort of data structure holding point correspondences

	def add_map_points(self, map_points, frame_idx):
		# Add a list of map_points
		self.map_points = self.map_points + map_points
		self.map_point_last_checked = self.map_point_last_checked + [frame_idx for _ in range(len(map_points))]
		over_count = len(self.map_points) - self.max_map_points
		if over_count > 0:
			new_idx = np.flip(np.argsort(self.map_point_last_checked))
			# self.map_points = [self.map_points[i] for i in new_idx]
			# self.map_point_last_checked = [self.map_point_last_checked[i] for i in new_idx]
			# self.map_points = [self.map_points[i] for i in range(over_count,len(self.map_points))]
			# self.map_point_last_checked = [self.map_point_last_checked[i] for i in range(over_count,len(self.map_point_last_checked))]
			idx = np.random.choice(len(self.map_points), self.max_map_points)
			self.map_points = [self.map_points[i] for i in idx]

	def add_frame(self, frame, pose, keyframe=False):
		self.frames.append(frame)
		self.camera_poses.append(pose)
		if keyframe:
			self.last_keyframe_idx = len(self.frames) - 1


	def update_keypoint_last_checked(self, points, frame_num):
		for idx in points:
			self.map_point_last_checked[idx] = frame_num

class SLAM():
	def __init__(self, match_descriptors_func, n_local_map_points):
		self.local_map = Map(n_local_map_points)
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
		start_points, next_points = self.init_frame.keypoint_coords[:,pairs[:,0]], frame.keypoint_coords[:,pairs[:,1]]
		start_points, next_points = start_points[:-1,:], next_points[:-1,:]
		descriptors = self.init_frame.descriptors[pairs[:,0]]

		# Do the triangulation
		point_4d, R, t, mask = triangulate(start_points, next_points, frame.intrinsic_mat, scale)
		descriptors = descriptors[mask]

		# Compue the camera and point positions in the global frame
		mat = np.matmul(make_translation_matrix(t), homogenize_matrix(R)) # Maps from the new camera frame to the old camera frame
		old_mat = np.matmul(make_translation_matrix(self.init_pose.pos), homogenize_matrix(quat_to_mat(self.init_pose.quat)))
		total_mat = np.matmul(old_mat, mat)
		new_pos = total_mat[:,3]
		new_quat = mat_to_quat(unhomogenize_matrix(total_mat))
		new_pose = Pose(new_pos, new_quat, t=frame.t)
		points_global = local_xyz_to_global_xyz(new_pose, point_4d)
		map_points = [MapPoint(points_global[:,i], descriptors[i]) for i in range(len(descriptors))]

		# Compute the angle between the two frames for each point
		start_vecs = (points_global - self.init_pose.pos)[0:3]
		next_vecs = (points_global - new_pose.pos)[0:3]
		start_vecs = start_vecs / np.linalg.norm(start_vecs, axis=0)
		next_vecs = next_vecs / np.linalg.norm(next_vecs, axis=0)
		dprods = np.sum(np.multiply(start_vecs, next_vecs), axis=0)

		import matplotlib.pyplot as plt
		# plt.scatter(next_points[0], next_points[1], color="blue", s=20**2)
		# plt.scatter(local_xyz_to_uv(frame.intrinsic_mat, point_4d)[0], local_xyz_to_uv(frame.intrinsic_mat, point_4d)[1], color="red", s=8**2)
		# plt.scatter(global_xyz_to_uv(new_pose, frame.intrinsic_mat, points_global)[0], global_xyz_to_uv(new_pose, frame.intrinsic_mat, points_global)[1], color="green", s=2**2)
		# plt.show()
		# plt.scatter(start_points[0], start_points[1], color="blue", s=20**2)
		# plt.scatter(global_xyz_to_uv(self.init_pose, self.init_frame.intrinsic_mat, points_global)[0], global_xyz_to_uv(self.init_pose, self.init_frame.intrinsic_mat, points_global)[1], color="green", s=2**2)
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
				map_obj.add_map_points(map_points, frame.index)
				map_obj.add_frame(self.init_frame, self.init_pose, keyframe=True)
				map_obj.add_frame(frame, new_pose, keyframe=True)
			return

	def track_next_frame(self, frame):
		# Find map points visible in the current frame, by looking at the previous frame's pose.
		map_point_coords = np.array([point.pos.flatten() for point in self.local_map.map_points]).T
		local_coords = global_xyz_to_local_xyz(self.local_map.camera_poses[-1], map_point_coords)
		uv_points, idx = local_only_good_image_idx(frame.intrinsic_mat, frame.img.shape, local_coords)
		descriptors = np.array([point.descriptor for point in self.local_map.map_points])[idx]

		# Match frame keypoints with map points
		pairs = self.match_descriptors(frame.descriptors, descriptors)
		print("Num matches: %d" % len(pairs))
		frame_points, map_points = frame.keypoint_coords[:,pairs[:,0]], (map_point_coords[:,idx])[:,pairs[:,1]]
		frame_points, map_points = frame_points[:-1,:], map_points[:-1,:]
		self.local_map.update_keypoint_last_checked(pairs[:,1], frame.index)

		# Reshape according to this: https://stackoverflow.com/questions/33696082/error-using-solvepnpransac-function
		frame_points = frame_points.T.reshape(-1,1,2)
		map_points = map_points.T.reshape(-1,1,3)
		camera_mat_3x3 = frame.intrinsic_mat[0:3,0:3]

		# Estimate the last camera pose
		pos = self.local_map.camera_poses[-1].pos
		quat = self.local_map.camera_poses[-1].quat
		input_Rvec = mat_to_rot_vec(quat_to_mat(quat).T)
		input_tvec = -1 * unhomogenize_vectors(pos).flatten()

		# Actually find the camera position
		suc, R_vec, t_vec, mask = cv2.solvePnPRansac(map_points, frame_points, camera_mat_3x3, distCoeffs=None,
		                                             rvec=input_Rvec, tvec=input_tvec, useExtrinsicGuess=True,
		                                             iterationsCount=10000, reprojectionError=2.0, confidence=0.999,
		                                             flags=cv2.SOLVEPNP_ITERATIVE)
		print("solvePnPRansac success? %s" % suc)
		# R, t are the rotation and translation from the world frame to the camera frame
		# So in our pose scheme, we have to use their inverses
		pose_mat = np.matmul(homogenize_matrix(rot_vec_to_mat(R_vec)).T, make_translation_matrix(-1 * t_vec))
		pos = pose_mat[:,3]
		quat = mat_to_quat(unhomogenize_matrix(pose_mat))
		camera_pose = Pose(pos, quat, t=frame.t)

		# Local map update step
		# Check if change in pose between this frame and the last keyframe is large enough
		# If it is, triangulate, and add any new points to the local map
		if homogeneous_norm(camera_pose.pos - self.local_map.camera_poses[-1].pos) > 0.5:
			return False
		dist = homogeneous_norm(camera_pose.pos - self.local_map.camera_poses[self.local_map.last_keyframe_idx].pos)
		print("dist, %f" % dist)
		this_frame_keyframe = dist > 0.03
		if this_frame_keyframe:
			last_keyframe = self.local_map.frames[self.local_map.last_keyframe_idx]
			last_keyframe_pos = self.local_map.camera_poses[self.local_map.last_keyframe_idx]
			# For now, just triangulate all points, and don't worry about duplicates
			pairs = self.match_descriptors(last_keyframe.descriptors, frame.descriptors)
			start_points, next_points = last_keyframe.keypoint_coords[:,pairs[:,0]], frame.keypoint_coords[:,pairs[:,1]]
			start_points, next_points = start_points[:-1,:], next_points[:-1,:]
			descriptors = last_keyframe.descriptors[pairs[:,0]]
			point_4d, R, t, mask = triangulate(start_points, next_points, frame.intrinsic_mat, dist)
			descriptors = descriptors[mask]

			mat = np.matmul(make_translation_matrix(t), homogenize_matrix(R)) # Maps from the new camera frame to the old camera frame
			old_mat = np.matmul(make_translation_matrix(last_keyframe_pos.pos), homogenize_matrix(quat_to_mat(last_keyframe_pos.quat)))
			total_mat = np.matmul(old_mat, mat)
			new_pos = total_mat[:,3]
			new_quat = mat_to_quat(unhomogenize_matrix(total_mat))
			# camera_pose = Pose(new_pos, new_quat, t=frame.t)

			points_global = local_xyz_to_global_xyz(camera_pose, point_4d)
			map_points = [MapPoint(points_global[:,i], descriptors[i]) for i in range(len(descriptors))]

			all_map_descriptors = [point.descriptor for point in self.local_map.map_points]
			pairs = self.match_descriptors(descriptors, all_map_descriptors)
			if len(pairs) == 0:
				no_duplicate_map_points = map_points
			else:
				duplicates = pairs[:,0]
				no_duplicate_map_points = [map_points[i] for i in range(len(map_points)) if i not in duplicates]
				print("%d duplicates" % len(duplicates))

		for map_obj in [self.local_map, self.global_map]:
			map_obj.add_frame(frame, camera_pose, keyframe=this_frame_keyframe)
			if this_frame_keyframe:
				print("Adding %d map points" % len(no_duplicate_map_points))
				map_obj.add_map_points(no_duplicate_map_points, frame.index)
		return True