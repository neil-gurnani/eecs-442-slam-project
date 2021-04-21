import numpy as np
import scipy, scipy.spatial
import matplotlib.pyplot as plt

from odometry_classes import MapPoint, Frame, Map, SLAM
from geometry import *
from dataloader import Dataloader
import fake_features

dataset_name = "plant_1"
data = Dataloader(dataset_name)

def fake_match(desc1, desc2):
	# Can maybe optimize this sort of function
	# https://stackoverflow.com/questions/49247506/how-to-efficiently-find-the-indices-of-matching-elements-in-two-lists
	return [(xi, xp) for (xi, x) in enumerate(desc1) for (xp, y) in enumerate(desc2) if x==y]

slam = SLAM(fake_match)

fake_points = fake_features.create_random_fake_map_points(data.image_groundtruths[0], 5)
# map_points_vec = np.array([fake_point.pos[:,0] for fake_point in fake_points]).T
# map_points_vec = np.concatenate([fake_point.pos for fake_point in fake_points], axis=1)
# descriptors_vec = np.array([fake_point.descriptor for fake_point in fake_points])

def fake_create_input_frame(index, fake_points):
	img = data.images[index]
	pose = data.image_groundtruths[index]
	fake_points_coords = np.concatenate([fake_point.pos for fake_point in fake_points], axis=1)
	fake_points_descriptors = np.array([fake_point.descriptor for fake_point in fake_points])
	local_points = global_xyz_to_local_xyz(pose, fake_points_coords)
	idx1 = only_positive_z_idx(local_points)
	uv_points = local_xyz_to_uv(data.intrinsic_mat, local_points[:,idx1])
	idx2 = only_within_image_idx(img.shape, uv_points)
	final_points = uv_points[:,idx2]
	final_descriptors = (fake_points_descriptors[idx1])[idx2]
	return Frame(img, final_points, final_descriptors)

# img0_points = only_within_image(data.images[0].shape, local_xyz_to_uv(data.intrinsic_mat, only_positive_z(global_xyz_to_local_xyz(data.image_groundtruths[0], map_points_vec))))
# img20_points = only_within_image(data.images[20].shape, local_xyz_to_uv(data.intrinsic_mat, only_positive_z(global_xyz_to_local_xyz(data.image_groundtruths[20], map_points_vec))))

f0 = fake_create_input_frame(0, fake_points)
f20 = fake_create_input_frame(20, fake_points)

fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(data.images[0])
# ax1.scatter(img0_points[0], img0_points[1])
ax1.imshow(f0.img)
ax1.scatter(f0.keypoints[0], f0.keypoints[1], c=f0.descriptors, cmap=plt.get_cmap("Set1"))
# ax2.imshow(data.images[20])
# ax2.scatter(img20_points[0], img20_points[1])
ax2.imshow(f20.img)
ax2.scatter(f20.keypoints[0], f20.keypoints[1], c=f20.descriptors, cmap=plt.get_cmap("Set1"))
plt.show()