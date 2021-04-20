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