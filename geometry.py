import numpy as np
import scipy, scipy.spatial
import cv2

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

	def __repr__(self):
		return "<Pose pos:%s quat:%s t:%s>" % (self.pos.flatten()[:-1], self.quat, self.t)

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

def rot_vec_to_mat(rot_vec):
	# Converts a 3x1 rotation vector to a 3x3 rotation matrix
	return scipy.spatial.transform.Rotation.from_rotvec(rot_vec.flatten()).as_matrix()

def rot_vec_to_quat(rot_vec):
	# Converts a 3x1 rotation vector to a quaternion w+xi+yj+zk stored as [x,y,z,w]
	return scipy.spatial.transform.Rotation.from_rotvec(rot_vec.flatten()).as_quat()

def homogenize_matrix(mat):
	# Converts a 3x3 matrix to a 4x4 homogeneous matrix
	new_mat = np.zeros((4,4))
	new_mat[0:3,0:3] = mat
	new_mat[3,3] = 1
	return new_mat

def unhomogenize_matrix(mat):
	# Converts a 4x4 homogeneous matrix to a 3x3 matrix
	return mat[0:3,0:3]

def homogenize_vectors(vecs):
	# Converts a 3xn set of vectors to a 4xn set of homogeneous vectors
	return np.pad(vecs, pad_width=((0,1),(0,0)), mode="constant", constant_values=1)

def unhomogenize_vectors(vecs):
	# Converts a 4xn set of homogeneous vectors to a 3xn set of vectors
	return vecs[0:3,:]

def homogeneous_norm(vec):
	# Returns the l2-norm of a 4x1 or 3x1 homogeneous vector
	return np.linalg.norm(vec.flatten()[:-1])

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
	xyz[0:3] = xyz[0:3] / xyz[2]
	return np.matmul(intrinsic_mat, xyz)

def local_xyz_to_global_xyz(camera_pose, xyz):
	# Transforms 3d points from the world frame to the camera frame
	# Camera poses are given as transformations from local to global frame
	# See: https://www.eth3d.net/slam_documentation
	# xyz should be a 4x1 set of homogeneous vectors in the world frame
	# Will return a 4x1 set of homogeneous vectors in the camera frame
	rot_mat = homogenize_matrix(quat_to_mat(camera_pose.quat))
	trans_mat = make_translation_matrix(camera_pose.pos)
	full_mat = np.matmul(trans_mat, rot_mat) # Rotate, then translate
	return np.matmul(full_mat, xyz)

def global_xyz_to_local_xyz(camera_pose, xyz):
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

def only_positive_z_idx(xyz):
	# Returns a list of the indices of only the points in xyz with positive z values
	# Often called before local_xyz_to_uv, to ensure we don't project points behind the camera
	# xyz should be a 4xn set of homogeneous vectors
	# Will return a list of n-k indices, corresponding to points in xyz with positive z values
	return xyz[2] > 0

def only_positive_z(xyz):
	# Returns a list of only the points in xyz with positive z values
	# Often called before local_xyz_to_uv, to ensure we don't project points behind the camera
	# xyz should be a 4xn set of homogeneous vectors
	# Will return a 4x(n-k) set of homogeneous vectors
	return xyz[:,only_positive_z_idx(xyz)]

def only_within_image_idx(image_shape, uv):
	# Returns a list of the indices of only the points in uv within the image bounds
	# The image convenion is that (0,0) is the center of the top left pixel
	# image_shape should be a tuple (rows, columns), i.e., (y, x)
	# uv should be a 3x1 set of homogeneous 2D vectors
	return np.logical_and.reduce((uv[0] >= 0, uv[0] <= image_shape[1], uv[1] >= 0, uv[1] <= image_shape[0]))

def only_within_image(image_shape, uv):
	# Returns a list of only the points in uv within the image bounds
	# The image convenion is that (0,0) is the center of the top left pixel
	# image_shape should be a tuple (rows, columns), i.e., (y, x)
	# uv should be a 3x1 set of homogeneous 2D vectors
	return uv[:,only_within_image_idx(image_shape, uv)]

def local_only_good_image_idx(intrinsic_mat, image_shape, xyz):
	# Given the camera intrinsic matrix (3x4), the image shape (with the convention that
	# (0,0) is the center of the top-left pixel), as a tuple (rows, columns), i.e., (y,x),
	# and xyz a 4xn set of homogenous vectors in the local frame.
	# Return the uv coordinates of only those points with positive z that project within the image
	# boundaries, and the corresponding indices in the original list
	idx1 = only_positive_z_idx(xyz)
	uv = local_xyz_to_uv(intrinsic_mat, xyz)
	idx2 = only_within_image_idx(image_shape, uv)
	mask = np.logical_and(idx1, idx2)
	return uv[:,mask], np.where(mask)[0]
	# NOTE: This could probably be optimized, so invalid points aren't projeced and such

def triangulate(old_points, new_points, camera_mat):
	# old_points and new_points should be 2xn sets of vectors
	# camera_mat should be the intrinsic matrix, either 3x3 or 3x4
	# Returns the 3d points as a 4xn set of homogeneous vectors in the coordinate frame of the new image
	# Returns the rotation and translation to get from the old camera pose to the new camera pose
	# Returns the mask of inlier points as a boolean array
	# See: https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
	if camera_mat.shape == (3,3):
		camera_mat_3x3 = camera_mat
	else:
		camera_mat_3x3 = camera_mat[0:3,0:3]
	new_points_norm = cv2.undistortPoints(new_points, cameraMatrix=camera_mat_3x3, distCoeffs=None)
	old_points_norm = cv2.undistortPoints(old_points, cameraMatrix=camera_mat_3x3, distCoeffs=None)

	# Get essential matrix and filter out false matches
	mat, mask = cv2.findEssentialMat(new_points_norm, old_points_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.001)
	mask_bool = mask.astype(bool).flatten()
	# print("Discarding %d points." % (len(old_points.T) - np.sum(mask_bool)))
	good_new_points = new_points[:,mask_bool]
	good_new_points_norm = new_points_norm[mask_bool,:,:]
	good_old_points = old_points[:,mask_bool]
	good_old_points_norm = old_points_norm[mask_bool,:,:]

	# Recover camera pose
	points, R, t, mask = cv2.recoverPose(mat, good_new_points_norm, good_old_points_norm)
	M_new = np.hstack((R, t))
	M_old = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
	P_new = np.dot(camera_mat_3x3,  M_new)
	P_old = np.dot(camera_mat_3x3,  M_old)

	# Triangulate in 3D
	point_4d_hom = cv2.triangulatePoints(P_old, P_new, good_old_points, good_new_points)
	point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))

	return point_4d, R, t, mask_bool