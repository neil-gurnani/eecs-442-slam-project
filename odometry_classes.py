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

class Frame():
	def __init__(self, img, keypoints, descriptors, intrinsic_mat):
		self.img = img
		self.keypoints = keypoints # Should be a 3x1 set of homogeneous 2D vectors (in the image plane)
		                           # By convention, the center of the top-left corner pixel is (0,0)
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints
		self.intrinsic_mat = intrinsic_mat

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

	def try_finish_initialization(self, frame):
		# See: https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
		camera_mat_3x3 = frame.intrinsic_mat[0:3,0:3]

		# Get possible matches
		pairs = self.match_descriptors(self.init_frame.descriptors, frame.descriptors)
		start_points, next_points = self.init_frame.keypoints[:,pairs[:,0]], frame.keypoints[:,pairs[:,1]]
		start_points, next_points = start_points[:-1,:], next_points[:-1,:]

		# Normalize
		start_points_norm = cv2.undistortPoints(start_points, cameraMatrix=camera_mat_3x3, distCoeffs=None)
		next_points_norm = cv2.undistortPoints(next_points, cameraMatrix=camera_mat_3x3, distCoeffs=None)

		# Get essential matrix and filter out false matches
		mat, mask = cv2.findEssentialMat(start_points_norm, next_points_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.99999, threshold=0.1)
		mask_bool = mask.astype(bool).flatten()
		good_start_points = start_points[:,mask_bool]
		good_start_points_norm = start_points_norm[mask_bool,:,:]
		good_next_points = next_points[:,mask_bool]
		good_next_points_norm = next_points_norm[mask_bool,:,:]

		# Recover camera pose
		points, R, t, mask = cv2.recoverPose(mat, good_start_points_norm, good_next_points_norm)
		M_next = np.hstack((R, t))
		M_start = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
		P_next = np.dot(camera_mat_3x3,  M_next)
		P_start = np.dot(camera_mat_3x3,  M_start)

		# Triangulate in 3D
		point_4d_hom = cv2.triangulatePoints(P_start, P_next, good_start_points, good_next_points)
		point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))

		import matplotlib.pyplot as plt
		plt.scatter(start_points[0], start_points[1], color="blue", s=20**2)
		plt.scatter(local_xyz_to_uv(frame.intrinsic_mat, point_4d)[0], local_xyz_to_uv(frame.intrinsic_mat, point_4d)[1], color="red")
		plt.show()

	def tracking_phase(self, frame):
		pass #TODO