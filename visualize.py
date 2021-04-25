"""Visualize file."""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import geometry
from geometry import Pose
from odometry_classes import Map, MapPoint, Frame, SLAM

"""
This code should be able to run without errors on Mac.
To run on windows, an X Ming server must be set up:
https://code.luasoftware.com/tutorials/wsl/show-matplotlib-in-wsl-ubuntu/
Run command: 
export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0
"""

def visualize(map_obj, ax):
    # A map object will be passed in that contains all data needed
    # map class is in odometry_classes

    # Creating the figure
    #fig = plt.figure(figsize = (10, 7))
    #ax = plt.axes(projection="3d")
    ax.clear()
    # add camera position
    if len(map_obj.camera_poses) > 0:
        camera_pose = map_obj.camera_poses[-1]
        cam_x, cam_y, cam_z = geometry.unhomogenize_matrix(camera_pose.pos)
    ax.scatter3D(cam_x, cam_y, cam_z, color="red", label="camera")

    # Plot the direction of the camera
    cam_quat = camera_pose.quat
    trans_mat = geometry.quat_to_mat(cam_quat)

    # Try having it in the form [X Y Z U V W]
    #unit_vec = np.array([1, 1, 1])
    #new_dir = np.matmul(trans_mat, unit_vec)
    #X,Y,Z = cam_x[0], cam_y[0], cam_z[0]
    #U,V,W = new_dir[0], new_dir[1], new_dir[2]
    #ax.quiver(X,Y,Z,U,V,W)
    ##print(cam_z)
    # add coordinate axis
    #X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],0.5, 0, 0
    #ax.quiver(X,Y,Z,U,V,W, color='blue')
    #X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],0,0.5,0
    ##ax.quiver(X,Y,Z,U,V,W, color='blue')
    #X,Y,Z,U,V,W = cam_x[0],cam_y[0],cam_z[0],0,0,0.5
    #print(W == cam_z[0]-0.5)
    #print(W)
    #print(cam_z[0]-0.5)
    #print(cam_z)
    #print("(%f, %f, %f) -> (%f, %f, %f)" % (X,Y,Z,U,V,W))
    #ax.quiver(X,Y,Z,U,V,W, color='blue')

    # Rotate by the quaternion
    # Example from stack overflow, RPR' is what we wnat
    # ham(ham(R,P),R'')
    #P = np.array([0, 1, 0, 0])
    #R = np.array([0.707, 0.0, 0.707, 0.0])
    #Rprime = np.array([0.707, 0.0, -.707, 0.0])

    def ham_prod(a,b):
        # Returns the hamilton product in form [w,x,y,z]
        # Used for multiplication of 2 quaternions
        a1, a2 = a[0], b[0]
        b1, b2 = a[1], b[1]
        c1, c2 = a[2], b[2]
        d1, d2 = a[3], b[3]

        elt1 = a1*a2 - b1*b2 - c1*c2 - d1*d2
        elt2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        elt3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        elt4 = a1*d2 + b1*c2 - c1*b2 + d1*a2

        return np.array([elt1, elt2, elt3, elt4])

    # Rotate coordinate axis by Camera Quaternion
    temp = camera_pose.quat
    cam_quat = np.array([temp[1], temp[2], temp[3], temp[0]])
    cam_quat_prime = [cam_quat[0], -cam_quat[1], -cam_quat[2], cam_quat[3]]
    xax = np.array([0,1,0,0])
    yax = np.array([0,0,1,0])
    zax = np.array([0,0,0,1])

    x_rot = ham_prod(ham_prod(cam_quat, xax), cam_quat_prime)[1:]
    y_rot = ham_prod(ham_prod(cam_quat, yax), cam_quat_prime)[1:]
    z_rot = ham_prod(ham_prod(cam_quat, zax), cam_quat_prime)[1:]

    # normalize the rotated axes (multiplied denom by 2 due to scale)
    x_rot /= 2*np.linalg.norm(x_rot)
    y_rot /= 2*np.linalg.norm(y_rot)
    z_rot /= 2*np.linalg.norm(z_rot)
    print("X rot: ", x_rot)
    print("Y rot: ", y_rot)
    print("Z rot: ", z_rot)
    X,Y,Z,U,V,W = cam_x[0], cam_y[0], cam_z[0], x_rot[0], x_rot[1], x_rot[2]
    ax.quiver(X,Y,Z,U,W,V, color='blue')
    X,Y,Z,U,V,W = cam_x[0], cam_y[0], cam_z[0], y_rot[0], y_rot[1], y_rot[2]
    ax.quiver(X,Y,Z,U,W,V, color='blue')
    X,Y,Z,U,V,W = cam_x[0], cam_y[0], cam_z[0], z_rot[0], z_rot[1], z_rot[2]
    ax.quiver(X,Y,Z,U,W,V, color='blue')

    # Plot the camera path
    for i in range(len(map_obj.camera_poses)-1):
        cur_pose = geometry.unhomogenize_matrix(map_obj.camera_poses[i].pos)[:3]
        next_pose = geometry.unhomogenize_matrix(map_obj.camera_poses[i+1].pos)[:3]
        x,y,z = cur_pose
        u,v,w = next_pose
        ax.plot3D([x[0],u[0]], [y[0],v[0]],[z[0],w[0]], color='black')


    # add each feature to plot
    x_coords, y_coords, z_coords = [], [], []
    for point in map_obj.map_points:
        px, py, pz = point.pos[0:3]
        x_coords.append(px)
        y_coords.append(py)
        z_coords.append(pz)
    
    ax.scatter3D(x_coords,y_coords,z_coords, color="green", label="feature")

    # Create title and legend
    ax.set_title("Display Camera pose")
    ax.legend(loc="upper right")


    # Name the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # find max & min
    xmin, xmax = cam_x[0]-2, cam_x[0]+2
    ymin, ymax = cam_y[0]-2, cam_y[0]+2
    zmin, zmax = cam_z[0]-2, cam_z[0]+2
    #print("Minx %d, Max %d" %(xmin, xmax)) (-2,1)
    ##print("Miny %d, Max %d" %(ymin, ymax)) (-2,1)
    #print("Minz %d, Max %d" %(zmin, zmax)) (0,3)
    # Set scale                  ----> EDIT THIS
    ax.set_xlim(-2,1)
    ax.set_ylim(-2,1)
    ax.set_zlim(0,3)
    # Annotate the camera
    #ax.text(cam_x[0],cam_y[0],cam_z[0], "Camera")

    # show plot
    plt.pause(0.25)
    print("Showing plot")
