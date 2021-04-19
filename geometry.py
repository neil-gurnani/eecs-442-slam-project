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

def homogenize_vectors(vecs):
	# Converts a 3xn set of vectors to a 4xn set of homogeneous vectors
	return np.pad(vecs, pad_width=((0,1),(0,0)), mode="constant", constant_values=1)

def unhomogenize_vectors(vecs):
	# Converts a 4xn set of homogeneous vectors to a 3xn set of vectors
	return vecs[0:3,:]

def make_translation_matrix(vec):
	# Converts a 3x1 vector or 4x1 homogeneous vector to a 4x4 homogeneous translation matrix
	mat = np.eye(4)
	mat[0:3,3] = vec
	return mat