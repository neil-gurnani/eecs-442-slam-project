import numpy as np
import cv2

from geometry import *

class MapPoint():
	def __init__(self, pos, descriptor):
		pos = np.array(pos).reshape(-1,1)
		if len(pos) == 3:
			pos = homogenize_vectors(pos)
		self.pos = pos # Should be a homogeneous 4x1 column vector
		self.descriptor = descriptor

class Frame():
	def __init__(self, img, keypoints, descriptors):
		self.img = img
		self.keypoints = keypoints # Should be a 3x1 set of homogeneous 2D vectors (in the image plane)
		                           # By convention, the center of the top-left corner pixel is (0,0)
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints

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
		pairs = self.match_descriptors(self.init_frame.descriptors, frame.descriptors)
		start_points, next_points = self.init_frame.keypoints[:,pairs[:,0]], frame.keypoints[:,pairs[:,1]]
		start_points, next_points = start_points[:-1].T, next_points[:-1].T # Make them lists of non-homogeneous vectors
		mat, mask = cv2.findHomography(start_points, next_points, cv2.RANSAC, 25.0)
		good_start_points = start_points[mask.astype(bool).flatten()]
		good_next_points = next_points[mask.astype(bool).flatten()]

	def tracking_phase(self, frame):
		pass #TODO