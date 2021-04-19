import numpy as np

from geometry import Pose

class MapPoint():
	def __init__(self, pose, descriptor):
		self.pose = pose             # Datatype should be Pose
		self.descriptor = descriptor

class Frame():
	def __init__(self):
		pass #TODO

class Map():
	def __init__(self):
		pass #TODO