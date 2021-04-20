"""Visualize file."""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from geometry import Pose
from odometry_classes import MapPoint

"""
This code should be able to run without errors on Mac.
To run on windows, an X Ming server must be set up:
https://code.luasoftware.com/tutorials/wsl/show-matplotlib-in-wsl-ubuntu/
"""

def visualize(camera_pose, features_list):
    # Pass in camera_pose as pose object and
    # a list of features which are Points 
    camera_pose = np.array([3, 3, 3])
    
    # Creating the figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection="3d")

    # add camera position
    cam_x, cam_y, cam_x = camera_pose.pos[0:3]
    ax.scatter3D(cam_x, cam_y, cam_z, color="red", label="camera")

    # add each feature to plot
    x_coords, y_cooords, z_coords = [], [], []
    for point in features_list:
        px, py, pz = point.pos[0:3]
        x_coords.append(px)
        y_coords.append(py)
        z_coords.append(pz)
    
    ax.scatter3D(x_coords,y_coords,z_coords, color="green", label="feature")
    
    # Create title and legend
    plt.title("Display Camera pose")
    plt.legend(loc="upper right")

    # Annotate the camera
    ax.text(40,40,50, "Camera")

    # show plot
    plt.show()
    print("Showing plot")
