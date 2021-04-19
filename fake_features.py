import numpy as np
import scipy

from geometry import *
from odometry_classes import MapPoint, Frame, Map

# This script is used to create fake features

def create_random_fake_features(camera_pose, count, random_range=(1,1,1), z_min=0.1):
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	# Returns random 3d points in the world frame, located in front of the given camera pose
	# The output, points_global, will be a 4xn set of homogeneous vectors
	lows = (-random_range[0], -random_range[1], z_min) # Can't have negative z
	highs = (random_range[0], random_range[1], random_range[2])
	points_local = homogenize_vectors(np.random.uniform(low=lows, high=highs, size=(count,3)).T)
	points_global = local_xyz_to_global_xyz(camera_pose, points_local)
	return points_global