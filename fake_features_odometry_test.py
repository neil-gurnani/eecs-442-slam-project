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

fake_points = fake_features.create_random_fake_map_points(data.image_groundtruths[0], 25)
map_points_vec = np.array([fake_point.pos[:,0] for fake_point in fake_points]).T
descriptors_vec = np.array([fake_point.descriptor for fake_point in fake_points])
img_points = local_xyz_to_uv(data.intrinsic_mat, global_xyz_to_local_xyz(data.image_groundtruths[0], map_points_vec))

plt.imshow(data.images[0])
plt.scatter(img_points[0], img_points[1])
plt.show()