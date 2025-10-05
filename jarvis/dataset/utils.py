"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import numpy as np
import matplotlib.pyplot as plt
import os,sys,inspect
import mpl_toolkits.mplot3d as mplot3d
import itertools
import cv2
import torch


class ReprojectionTool:
    def __init__(self, root_dir, calib_paths):
        self.cameras = {}
        for camera in calib_paths:
            self.cameras[camera] = Camera(camera,
                        os.path.join(root_dir, calib_paths[camera]))
        self.camera_list = [self.cameras[cam] for cam in self.cameras]
        self.num_cameras = len(self.camera_list)
        # Only store 3x4 camera matrices for DLT projection
        self.cameraMatrices = torch.zeros(self.num_cameras, 4,3)
        for i,cam in enumerate(self.cameras):
            self.cameraMatrices[i] =  torch.from_numpy(
                        self.cameras[cam].cameraMatrix).transpose(0,1)


    def reprojectPoint(self,point3D):
        pointsRepro = np.zeros((self.num_cameras, 2))
        for i,cam in enumerate(self.camera_list):
            # Simple DLT projection using 3x4 camera matrix
            pointRepro = cam.cameraMatrix.dot(
                        np.concatenate((point3D, np.array([1]))))
            # Normalize by homogeneous coordinate
            pointRepro = (pointRepro/pointRepro[-1])[:2]
            pointsRepro[i] = pointRepro
        return pointsRepro


    def reconstructPoint(self,points, camsToUse = None):
        if camsToUse == None:
            camsToUse = range(len(self.cameras))
        if (len(camsToUse) > 1):
            camMats = []
            for i,camera in enumerate(self.cameras):
                if i in camsToUse:
                    cam = self.cameras[camera]
                    camMats.append(cam.cameraMatrix)

            pointsToUse = np.zeros((2, len(camsToUse)))
            for i,cam in enumerate(camsToUse):
                # Use points directly without distortion correction
                pointsToUse[:,i] = points[:,cam]
            
            # DLT triangulation: build matrix A from point-camera ray constraints
            # For each camera: x_i × (P_i * X) = 0, which gives 2 equations per camera
            A = np.zeros((pointsToUse.shape[1]*2, 4))
            for i in range(pointsToUse.shape[1]):
                A[2*i:2*i+2] = pointsToUse[:, i].reshape(2,1).dot(
                            camMats[i][2].reshape(1,4)) - camMats[i][0:2]
            _,_,vh = np.linalg.svd(A)
            V = np.transpose(vh)
            X = V[:,-1]
            X = X/X[-1]
            X = X[0:3]
            return X
        else:
            return np.array([0,0,0])


class Camera:
    def __init__(self, name, calib_path):
        self.name = name
        print(self.name)
        # Load only the 3x4 projection matrix for DLT
        self.cameraMatrix = self.get_mat_from_file(calib_path,
                    'projectionMatrix')

    def get_mat_from_file(self, filepath, nodeName):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        return fs.getNode(nodeName).mat()