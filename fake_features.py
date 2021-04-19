import numpy as np
import scipy

from geometry import *
from odometry_classes import MapPoint, Frame, Map

# This script is used to create fake features

def create_random_fake_features(camera_pose, count, random_range=(1,1,1), z_min=0.1):
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	rot_mat = homogenize_matrix(quat_to_mat(camera_pose.quat))
	trans_mat = make_translation_matrix(camera_pose.pos)
	full_mat = np.matmul(trans_mat, rot_mat) # Rotate, then translate
	lows = (-random_range[0], -random_range[1], z_min) # Can't have negative z
	highs = (random_range[0], random_range[1], random_range[2])
	points_local = homogenize_vectors(np.random.uniform(low=lows, high=highs, size=(count,3)).T)
	points_global = np.matmul(full_mat, points_local)
	return points_global

def project_fake_feature_to_image(feature):
	pass