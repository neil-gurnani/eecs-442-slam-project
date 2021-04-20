import numpy as np

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
		self.keypoints = keypoints # Should be a list of 2d points in the image.
		                           # By convention, the center of the top-left corner pixel is (0,0)
		self.descriptors = descriptors # Should be in 1-1 correspondence with keypoints

class Map():
	def __init__(self):
		self.frames = []
		self.map_points = []
		# Need some sort of data structure holding point correspondences

class Odometry():
	def __init__(self):
		self.local_map = Map()
		self.global_map = Map()
		self.has_started_initialization
		self.has_finished_initialization

	def next_frame(self, frame):
		if self.has_finished_initialization:
			self.tracking_phase(frame)
		elif self.has_started_initialization:
			self.has_finished_initialization = self.try_finish_initialization(frame)
		else:
			self.start_initialization(frame)

	def start_initialization(self, frame):
		pass #TODO

	def try_finish_initialization(self, frame):
		pass #TODO

	def tracking_phase(self, frame):
		pass #TODO