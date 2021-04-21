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
	return np.array([(xi, xp) for (xi, x) in enumerate(desc1) for (xp, y) in enumerate(desc2) if x==y])

slam = SLAM(fake_match)

fake_points = fake_features.create_random_fake_map_points(data.image_groundtruths[0], 15)

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

frames = []
n_images = len(data.images)
for i in range(n_images):
	frames.append(fake_create_input_frame(i, fake_points))

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(frames[0].img)
# ax1.scatter(frames[0].keypoints[0], frames[0].keypoints[1], c=frames[0].descriptors, cmap=plt.get_cmap("Set1"))
# for i in range(n_images):
# 	ax2.cla()
# 	ax2.imshow(frames[i].img)
# 	ax2.scatter(frames[i].keypoints[0], frames[i].keypoints[1], c=frames[i].descriptors, cmap=plt.get_cmap("Set1"))
# 	plt.pause(0.1)

slam.start_initialization(frames[0], data.image_groundtruths[0])
slam.try_finish_initialization(frames[20])
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(frames[0].img)
ax1.scatter(frames[0].keypoints[0], frames[0].keypoints[1], c=frames[0].descriptors, cmap=plt.get_cmap("Set1"))
ax2.imshow(frames[20].img)
ax2.scatter(frames[20].keypoints[0], frames[20].keypoints[1], c=frames[20].descriptors, cmap=plt.get_cmap("Set1"))
plt.show()