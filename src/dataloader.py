import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math

from geometry import Pose

def get_list_of_datasets(data_dir="./data"):
	return os.listdir(data_dir)

class Dataloader():
	def __init__(self, dataset_name, data_dir="./data"):
		self.data_dir = data_dir
		self.dataset_name = dataset_name
		self.image_folder_path = self.data_dir + "/" + self.dataset_name + "/rgb/"
		self.calibration_filename = self.data_dir + "/" + self.dataset_name + "/calibration.txt"
		self.groundtruth_filename = self.data_dir + "/" + self.dataset_name + "/groundtruth.txt"

		# Load the calibration data
		with open(self.calibration_filename) as f:
			raw_calibration = [float(s) for s in f.read().split(" ")]
		f.close()
		self.fx, self.fy, self.cx, self.cy = raw_calibration
		self.f = np.array([self.fx, self.fy])
		self.c = np.array([self.cx, self.cy])

		# Create intrinsic matrix (stored as a 3x4 matrix)
		self.intrinsic_mat = np.array([
			[self.fx, 0, self.cx, 0],
			[0, self.fy, self.cy, 0],
			[0, 0, 1, 0]])

		# Load the groundtruth timestamps
		self.groundtruths = []
		self.groundtruth_timestamps = []
		with open(self.groundtruth_filename) as f:
			for line in f:
				if line[0] != "#":
					raw_line = [float(s) for s in line.split(" ")]
					t, tx, ty, tz, qx, qy, qz, qw = raw_line
					self.groundtruths.append(Pose([tx, ty, tz], [qx, qy, qz, qw], t))
					self.groundtruth_timestamps.append(t)
		self.groundtruth_timestamps = np.array(self.groundtruth_timestamps)

		# Load the images
		print("Loading images...")
		fnames = os.listdir(self.image_folder_path)
		fnames.sort()
		self.images = []
		self.image_timestamps = []
		for image_fname in tqdm(fnames):
			image = plt.imread(self.image_folder_path + image_fname)
			self.images.append(image)
			self.image_timestamps.append(float(image_fname[:-4]))
		self.image_timestamps = np.array(self.image_timestamps)

		# Match image timestamps with closest ground truth timestamp
		# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
		self.image_groundtruths_idx = []
		self.image_groundtruths = []
		for timestamp in self.image_timestamps:
			idx = np.searchsorted(self.groundtruth_timestamps, timestamp, side="left")
			if idx > 0 and (idx == len(self.groundtruth_timestamps)
				or math.fabs(timestamp - self.groundtruth_timestamps[idx-1])
				< math.fabs(timestamp - self.groundtruth_timestamps[idx])):
				idx -= 1
			self.image_groundtruths_idx.append(idx)
			self.image_groundtruths.append(self.groundtruths[idx])