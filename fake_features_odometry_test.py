import numpy as np
import scipy, scipy.spatial
import matplotlib.pyplot as plt

from odometry_classes import MapPoint, Frame, Map, SLAM
from geometry import *
from dataloader import Dataloader
import fake_features

dataset_name = "plant_1"
data = Dataloader(dataset_name)

slam = SLAM()

fake_points = fake_features.create_random_fake_map_points(data.image_groundtruths[0], 5)
map_points_vec = np.array([fake_point.pos[:,0] for fake_point in fake_points]).T
descriptors_vec = np.array([fake_point.descriptor for fake_point in fake_points])

img0_points = only_within_image(data.images[0].shape, local_xyz_to_uv(data.intrinsic_mat, only_positive_z(global_xyz_to_local_xyz(data.image_groundtruths[0], map_points_vec))))
img20_points = only_within_image(data.images[20].shape, local_xyz_to_uv(data.intrinsic_mat, only_positive_z(global_xyz_to_local_xyz(data.image_groundtruths[20], map_points_vec))))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(data.images[0])
ax1.scatter(img0_points[0], img0_points[1])
ax2.imshow(data.images[20])
ax2.scatter(img20_points[0], img20_points[1])
plt.show()