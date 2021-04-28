import numpy as np
import scipy, scipy.spatial
import matplotlib.pyplot as plt
import cv2

from odometry_classes import MapPoint, Frame, Map, SLAM
from geometry import *
from dataloader import Dataloader

dataset_name = "table_3"
data = Dataloader(dataset_name)

orb = cv2.ORB_create(nfeatures=8000, edgeThreshold=0)
def process_frame(index, brightness_adjustment=2.0):
	img = np.array(cv2.cvtColor(data.images[index], cv2.COLOR_RGB2GRAY) * 255 * brightness_adjustment, dtype=np.uint8)
	kp, des = orb.detectAndCompute(img, None)
	return Frame(data.images[index], kp, des, data.intrinsic_mat, t=data.image_timestamps[index], index=index, use_opencv_keypoints=True)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
dist_thresh = 50.0
def match_descriptors(desc1, desc2):
	# print(len(desc1), len(desc2))
	matches = matcher.match(desc1, desc2)
	pairs = np.array([[match.queryIdx, match.trainIdx] for match in matches if match.distance <= dist_thresh])
	if len(pairs) == 0:
		return pairs
	pairs = pairs[np.unique(pairs[:,0], return_index=True)[1]]
	if len(pairs) == 0:
		return pairs
	pairs = pairs[np.unique(pairs[:,1], return_index=True)[1]]
	return pairs

# frame0 = process_frame(0)
# frame1 = process_frame(1)
# pairs = match_descriptors(frame0.descriptors, frame1.descriptors)
# import pdb
# pdb.set_trace

slam = SLAM(match_descriptors, 5000)

init_frame = process_frame(0)
slam.start_initialization(init_frame, data.image_groundtruths[0])

real_positions = []
est_positions = []
real_positions.append(data.image_groundtruths[0].pos.flatten()[:-1])
est_positions.append(data.image_groundtruths[0].pos.flatten()[:-1])

n_images = len(data.images)
fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(init_frame.img)
# ax1.scatter(init_frame.keypoint_coords[0], init_frame.keypoint_coords[1], s=2**2)
n_failures_in_a_row = 0
for i in range(1, n_images)[::3]:
	print("\nProcessing frame %d" % i)
	current_frame = process_frame(i)
	good = False
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
			real_positions.append(act_pose.pos.flatten()[:-1])
			est_positions.append(est_pose.pos.flatten()[:-1])
	else:
		good = slam.track_next_frame(current_frame)
		if good:
			est_pose = slam.global_map.camera_poses[-1]
			act_pose = data.image_groundtruths[i]
			# print("\t\t\t\t\t\t\t\t%s" % data.image_groundtruths_idx[i])
			pos_err = np.linalg.norm(homogeneous_norm(est_pose.pos - act_pose.pos))
			quat_err = quat_error(est_pose.quat, act_pose.quat)
			print("Position error: %f Orientation error: %f" % (pos_err, quat_err))
			real_positions.append(act_pose.pos.flatten()[:-1])
			est_positions.append(est_pose.pos.flatten()[:-1])
			n_failures_in_a_row = 0
		else:
			print("solvePnP failure (skipping).")
			n_failures_in_a_row += 1
		if good:
			ax1.cla()
			ax1.imshow(current_frame.img)
			ax1.scatter(current_frame.keypoint_coords[0], current_frame.keypoint_coords[1], s=2**2)
			ax2.cla()
			ax2.plot(np.array(real_positions)[:,0], np.array(real_positions)[:,1], color="orange", label="Ground Truth")
			ax2.plot(np.array(est_positions)[:,0], np.array(est_positions)[:,1], color="black", label="Estimated")
			ax2.legend()
			plt.pause(0.001)
	if n_failures_in_a_row > 10:
		break

real_positions = np.array(real_positions)
est_positions = np.array(est_positions)

# plt.figure()
# plt.plot(real_positions[:,0], real_positions[:,1])
# plt.plot(est_positions[:,0], est_positions[:,1])
plt.show()