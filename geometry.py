import numpy as np
import scipy, scipy.spatial

class Pose():
	def __init__(self, pos, quat, t=None):
		pos = np.array(pos).reshape(-1,1)
		if len(pos) == 3:
			pos = homogenize_vectors(pos)
		self.pos = pos # Should be a homogeneous 4x1 column vector
		self.quat = np.array(quat) / np.linalg.norm(quat) # Normalize to a unit quaternion.
		                                        # Format is [x,y,z,w] for the quaternion xi+yj+zk+w.
		                                        # This "scalar-last" convention is used by scipy.
		self.t = t # Timestamp is optional

def identity_pose():
	return Pose([0,0,0], identity_quat())

def identity_quat():
	return np.array([0,0,0,1])

def quat_to_mat(quat):
	# Converts a quaternion w+xi+yj+zk stored as [x,y,z,w] to a 3x3 rotation matrix
	return scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()

def mat_to_quat(mat):
	# Converts a 3x3 rotation matrix to a quaternion w+xi+yj+zk stored as [x,y,z,w]
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
	mat[0:3,3] = vec[0:3,0]
	return mat

def local_xyz_to_uv(intrinsic_mat, xyz):
	# Projects 3d points from the camera frame onto the image plane, via the
	# pinhole camera model.
	# Intrinsic_mat should be the 3x4 camera intrinsic matrix
	# xyz should be a 4x1 set of homogeneous vectors
	# Will return a 3x1 set of homogeneous 2D vectors (in the image plane)
	return np.matmul(intrinsic_mat, xyz)

def global_xyz_to_local_xyz(camera_pose, xyz):
	# Transforms 3d points from the world frame to the camera frame
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	# xyz should be a 4x1 set of homogeneous vectors in the world frame
	# Will return a 4x1 set of homogeneous vectors in the camera frame
	rot_mat = homogenize_matrix(quat_to_mat(camera_pose.quat))
	trans_mat = make_translation_matrix(camera_pose.pos)
	full_mat = np.matmul(trans_mat, rot_mat) # Rotate, then translate
	return np.matmul(full_mat, xyz)

def local_xyz_to_global_xyz(camera_pose, xyz):
	# Transforms 3d points from the camera frame to the world frame
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	# xyz should be a 4x1 set of homogeneous vectors in the camera frame
	# Will return a 4x1 set of homogeneous vectors in the world frame
	rot_mat = homogenize_matrix(quat_to_mat(camera_pose.quat)).T
	trans_mat = make_translation_matrix(-1 * camera_pose.pos)
	full_mat = np.matmul(rot_mat, trans_mat)
	return np.matmul(full_mat, xyz)

def global_xyz_to_uv(camera_pose, intrinsic_mat, xyz):
	# Transforms 3d points from the world frame to the camera frame, and then projects onto the image plane.
	# Intrinsic_mat shoul dbe the 3x4 camera intrinsic matrix
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	# xyz should be a 4x1 set of homogeneous vectors in the world frame
	# Will return a 3x1 set of homogeneous 2D vectors (in the image plane)
	return local_xyz_to_uv(intrinsic_mat, global_xyz_to_local_xyz(camera_pose, xyz))