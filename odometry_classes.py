import numpy as np

from geometry import *

class MapPoint():
	def __init__(self, pos, descriptor):
		pos = np.array(pos).reshape(-1,1)
		if len(pos) == 3:
			pos = homogenize_vectors(pos)
		self.pos = pos # Should be a homogeneous 4x1 column vector
		self.descriptor = descriptor

class Frame():
	def __init__(self):
		pass #TODO

class Map():
	def __init__(self):
		pass #TODO