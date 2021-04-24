import numpy as np
import scipy, scipy.spatial
import matplotlib.pyplot as plt
import cv2

from odometry_classes import MapPoint, Frame, Map, SLAM
from geometry import *
from dataloader import Dataloader

dataset_name = "plant_1"
data = Dataloader(dataset_name)

orb = cv2.ORB_create(nfeatures=10000)
def process_frame(index):
	img = np.array(cv2.cvtColor(data.images[index], cv2.COLOR_RGB2GRAY) * 255, dtype=np.uint8)
	kp, des = orb.detectAndCompute(img, None)
	return Frame(data.images[index], kp, des, data.intrinsic_mat, t=data.image_timestamps[index], index=index, use_opencv_keypoints=True)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
dist_thresh = 75.0
def match_descriptors(desc1, desc2):
	matches = matcher.match(desc1, desc2)
	pairs = np.array([[match.queryIdx, match.trainIdx] for match in matches if match.distance <= dist_thresh])
	return pairs

# frame0 = process_frame(0)
# frame1 = process_frame(1)
# pairs = match_descriptors(frame0.descriptors, frame1.descriptors)
# import pdb
# pdb.set_trace

slam = SLAM(match_descriptors)

init_frame = process_frame(0)
slam.start_initialization(init_frame, data.image_groundtruths[0])
n_images = len(data.images)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(init_frame.img)
ax1.scatter(init_frame.keypoint_coords[0], init_frame.keypoint_coords[1], s=2**2)
for i in range(2, n_images):
	print("\nProcessing frame %d" % i)
	current_frame = process_frame(i)
	ax2.cla()
	ax2.imshow(current_frame.img)
	ax2.scatter(current_frame.keypoint_coords[0], current_frame.keypoint_coords[1], s=2**2)
	plt.pause(0.25)
	if not slam.has_finished_initialization:
		scale = homogeneous_norm(data.image_groundtruths[0].pos - data.image_groundtruths[i].pos)
		slam.try_finish_initialization(current_frame, scale)
		if(slam.has_finished_initialization):
			est_pose = slam.global_map.camera_poses[-1]
			act_pose = data.image_groundtruths[i]
			pos_err = np.linalg.norm(homogeneous_norm(est_pose.pos - act_pose.pos))
			quat_err = quat_error(est_pose.quat, act_pose.quat)
			print("Position error: %f Orientation error: %f" % (pos_err, quat_err))
			print("Map has %d points" % len(slam.local_map.map_points))
	else:
		slam.track_next_frame(current_frame, slam.local_map.camera_poses[-1])
		est_pose = slam.global_map.camera_poses[-1]
		act_pose = data.image_groundtruths[i]
		pos_err = np.linalg.norm(homogeneous_norm(est_pose.pos - act_pose.pos))
		quat_err = quat_error(est_pose.quat, act_pose.quat)
		print("Position error: %f Orientation error: %f" % (pos_err, quat_err))