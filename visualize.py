"""Visualize file."""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import geometry
from geometry import Pose
from odometry_classes import Map, MapPoint, Frame

"""
This code should be able to run without errors on Mac.
To run on windows, an X Ming server must be set up:
https://code.luasoftware.com/tutorials/wsl/show-matplotlib-in-wsl-ubuntu/
Run command: 
export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0
"""

def visualize(map_obj):
    # A map object will be passed in that contains all data needed
    # map class is in odometry_classes

    # Creating the figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection="3d")

    # add camera position
    camera_pose = map_obj.self.camera_poses[-1]
    cam_x, cam_y, cam_z = geometry.unhomogenize_matrix(camera_pose.pos)
    ax.scatter3D(cam_x, cam_y, cam_z, color="red", label="camera")

    # Plot the direction of the camera
    cam_quat = camera_pose.quat
    trans_mat = geometry.quat_to_mat(cam_quat)

    # Try having it in the form [X Y Z U V W]
    unit_vec = np.array([1, 1, 1])
    new_dir = np.matmul(trans_mat, unit_vec)
    X,Y,Z = cam_x[0], cam_y[0], cam_z[0]
    U,V,W = new_dir[0], new_dir[1], new_dir[2]
    ax.quiver(X,Y,Z,U,V,W)

    # add coordinate axis
    X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],cam_x[0]+1,cam_y[0],cam_z[0]
    ax.quiver(X,Y,Z,U,V,W, color='black')
    X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],cam_x[0],cam_y[0]+1,cam_z[0]
    ax.quiver(X,Y,Z,U,V,W, color='black')
    X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],cam_x[0],cam_y[0],cam_z[0]+1
    ax.quiver(X,Y,Z,U,V,W, color='black')

    # Plot the camera path
    for i in range(len(map_obj.camera_poses)-1):
        cur_pose = map_obj.camera_pose[i]
        next_pose = map_obj.camera_pose[i+1]

        ax.plot3D(cur_pose, next_pose, color='red')


    # add each feature to plot
    x_coords, y_coords, z_coords = [], [], []
    for point in map_obj.map:
        px, py, pz = point.pos[0:3]
        x_coords.append(px)
        y_coords.append(py)
        z_coords.append(pz)
    
    ax.scatter3D(x_coords,y_coords,z_coords, color="green", label="feature")

    # Create title and legend
    plt.title("Display Camera pose")
    plt.legend(loc="upper right")


    # Name the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set scale                  ----> EDIT THIS
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    # Annotate the camera
    #ax.text(cam_x[0],cam_y[0],cam_z[0], "Camera")

    # show plot
    plt.show()
    print("Showing plot")
