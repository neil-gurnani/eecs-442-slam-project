import numpy as np

class Pose():
	def __init__(self, pos, quat, t=None):
		self.pos = pos
		self.quat = quat / np.linalg.norm(quat) # Normalize to a unit quaternion.
		                                        # Format is [x,y,z,w] for the quaternion xi+yj+zk+w.
		                                        # This "scalar-last" convention is used by scipy.
		self.t = t # Timestamp is optional