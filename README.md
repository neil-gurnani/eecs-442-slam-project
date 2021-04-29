# EECS 442 SLAM Final Project

EECS 442 Slam Final Project

## Overview

Our Computer Vision final project centered around Simultaneous localization and mapping (SLAM), which is a famous  problem in robotics,  where a computer  system  at-tempts to construct a map of its environment, while simul-taneously maintaining knowledge of its position within that map.

The goal of our project is to produce an effective SLAM system using only a single, ordinary (2D) camera for sensory input. This type of camera is ubiquitous in our modern, digital world, so our SLAM algorithm could easily be applied to almost any mobile device. 

The Final Project Report Specification can be found in [report specification](EECS_442_Computer_Vision_Spec.pdf)

A summary of our findings can be found in our [report](EECS_442_Computer_Vision_SLAM_Project.pdf) for [EECS 442: Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/) course at the University of Michigan during the Winter 2021 Semester.   You can view our latest report on [Overleaf](https://www.overleaf.com/project/608447d3ec80589a09854d0f).

Our source code for conducting SLAM and analyzing our data can be found in the **src** folder.


## Motivation
The motivation for this project came from our first team meeting, where we discovered a common interest in Autonomous Vehicles, as well as robotic systems that use advanced technniques such as SLAM. We came to the realization that SLAM has a vast number of real world practical applications across a growing variety of fields and disciplines. After a group conversation, we determined that this would be quite a fascinating project to move forward with to further explore our interest in this subdiscipline of Computer Vision.

## Data
We were able to obtain the data we needed for this project from the ETH3D dataset. With the video sequences and ground truth camera poses taken from ETH3D, we were able to both qualitatively and quantitatively analyze the quality of our results. :
* [ETH3D SLAM Datasets](https://www.eth3d.net/slam_datasets)

In order to fetch all of the data we collected, we simply build and run the following shell script: 
```bash
chmod +x download_dataset.sh 
./download_dataset.sh
```
Running this shell script would effectively import all the data from the ETH3D website to our created **data** folder.


## Dependencies
### Recommended
* [Anaconda 5.2 for Python 3.7.x](https://www.anaconda.com/download/)

### Required
* [Numpy](https://www.numpy.org)
    * conda install -c anaconda numpy
* [Matplotlib](https://matplotlib.org)
    * conda install -c conda-forge matplotlib
* [OpenCV](https://opencv.org)
    * conda install -c menpo opencv


## Useful Links and References
* [OpenCV Documentation on Feature Extraction and Matching](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)
* [OrbSLAM Monocular SLAM](https://openslam-org.github.io/orbslam.html)
* [Monoslam Real Time Camera SLAM](https://www.robots.ox.ac.uk/~lav/Papers/davison_etal_pami2007/davison_etal_pami2007.pdf)
* [Monocular SLAM for Visual Odometry](https://www.hindawi.com/journals/mpe/2012/676385/)
* [SURF, Speeded up Robust Features](https://people.ee.ethz.ch/~surf/eccv06.pdf)
* [Junction Tree Filters for SLAM](http://ai.stanford.edu/~paskin/slam/)
* [Distinctive Image Features from Scale Invariant Keypoints](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
* [Other Monocular Visual Odometry Github Implementation](https://github.com/felixchenfy/Monocular-Visual-Odometry)


