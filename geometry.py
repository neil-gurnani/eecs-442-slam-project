import numpy as np
import scipy, scipy.spatial

class Pose():
	def __init__(self, pos, quat, t=None):
		self.pos = pos # Should be a row vector
		self.quat = quat / np.linalg.norm(quat) # Normalize to a unit quaternion.
		                                        # Format is [x,y,z,w] for the quaternion xi+yj+zk+w.
		                                        # This "scalar-last" convention is used by scipy.
		self.t = t # Timestamp is optional

def quat_to_mat(quat):
	return scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()

def mat_to_quat(mat):
	return scipy.spatial.transform.Rotation.from_matrix(mat).as_quat()

def homogenize_matrix(mat):
	# Converts a 3x3 matrix to a 4x4 homogeneous matrix
	new_mat = np.zeros((4,4))
	new_mat[0:3,0:3] = mat
	new_mat[3,3] = 1
	return new_mat

def homogenize_vector(vec):
	# Converts a 3x1 vector to a 4x1 homogeneous vector
	new_vec = np.ones((4,1))
	new_vec[0:3,0] = vec
	return new_vec