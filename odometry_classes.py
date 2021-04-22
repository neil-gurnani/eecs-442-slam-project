import numpy as np
import cv2
import pdb

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
	def __init__(self, img, keypoints, descriptors, intrinsic_mat, t=None):
		self.img = img
		self.keypoints = keypoints # Should be a 3x1 set of homogeneous 2D vectors (in the image plane)
		                           # By convention, the center of the top-left corner pixel is (0,0)
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints
		self.intrinsic_mat = intrinsic_mat
		self.t = t

class Map():
	def __init__(self):
		self.frames = []
		self.camera_poses = []
		self.map_points = []
		# Need some sort of data structure holding point correspondences

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

	def next_frame(self, frame):
		if not self.has_finished_initialization:
			self.has_finished_initialization = self.try_finish_initialization(frame)
		else:
			self.start_initialization(frame)

	def try_finish_initialization(self, frame, scale):
		# Get possible matches
		pairs = self.match_descriptors(self.init_frame.descriptors, frame.descriptors)
		start_points, next_points = self.init_frame.keypoints[:,pairs[:,0]], frame.keypoints[:,pairs[:,1]]
		start_points, next_points = start_points[:-1,:], next_points[:-1,:]

		# Normalize
		point_4d, R, t, mask = triangulate(start_points, next_points, frame.intrinsic_mat)
		t = t * scale # Scale for the first displacement only

		mat = np.matmul(make_translation_matrix(t), homogenize_matrix(R)) # Maps from the old camera frame to the new camera frame
		new_pos = np.matmul(mat, self.init_pose.pos)
		new_quat = mat_to_quat(unhomogenize_matrix(np.matmul(mat, homogenize_matrix(quat_to_mat(self.init_pose.quat)))))
		new_pose = Pose(new_pos, new_quat, t=frame.t)
		# print(new_pose)
		# print(new_pos)
		# print(new_quat)
		point_global = local_xyz_to_global_xyz(new_pose, point_4d)

		# import matplotlib.pyplot as plt
		# plt.scatter(start_points[0], start_points[1], color="blue", s=20**2)
		# plt.scatter(local_xyz_to_uv(frame.intrinsic_mat, point_4d)[0], local_xyz_to_uv(frame.intrinsic_mat, point_4d)[1], color="red")
		# plt.show()

		return new_pos, new_quat, point_global # TEMPORARY

	def tracking_phase(self, frame):
		pass #TODO