#!/usr/bin/env python
# coding: utf-8

import cv2, PIL
import numpy as np
import math
import matplotlib.pyplot as plt
from cv2 import aruco

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def main():
    
        
    cap = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = cap.read()
        frame_copy = frame.copy();
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        #frame = cv2.imread("_data/girouette_south.png")
        
        # post processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if(ids is not None and ids[0] == 17):
            
            # pour avoir cameraMatrix, distCoeffs, il faut caliberer la caméra
            cameraMatrix = cv2.UMat(np.array(
                [[501.00981086, 0, 322.60954865],
                 [0, 502.06765583, 240.3226073],
                 [0, 0., 1.]]))
            distCoeffs = cv2.UMat(np.array([1.54271483e-01, -8.59578664e-02, 1.13392638e-03, 4.86769054e-04]))

            
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)


            cv2.aruco.drawDetectedMarkers(frame_copy, corners, ids)
            cv2.aruco.drawAxis(frame_copy, cameraMatrix, distCoeffs, rvecs, tvecs, 0.1)
            
            # Converts a rotation matrix to a rotation vector or vice versa.
            R = cv2.Rodrigues(rvecs)[0]

            # convert a rotation matrix to Euler angles
            e_m = rotationMatrixToEulerAngles(R.get())
            #print(e_m)

            e_m_deg = []
            # convert to rad -> deg
            for i in e_m:
                e_m_deg.append((i * 180) / math.pi)

            # check y coordinate
            if (e_m_deg[1] > 0):
                print("Nord")
            else:
                print("Sud")
        
        cv2.imshow('camera', frame_copy)
        #cap.release()
        #cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()