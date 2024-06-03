import numpy as np
import cv2
# from scipy import linalg

def find_fundamental_matrix(pts1, pts2, camera_matrix1, camera_matrix2, essential_matrix):
    pass

def find_essential_matrix(pts1, pts2):
    pass

def intrinsic_matrix():
    pass

def camera_matrix():
    X, Y, Z = int, int, int
    X_w, Y_w, Z_w = int, int , int
    f, px, py = int, int, int

    hom_img = np.array([X, Y, Z])
    hom_world = np.array([X, Y, Z, 1])

    # intrinsic K
    K = np.array([[f, 0, px, 0],
                  [0, f, py, 0],
                  [0, 0, 1, 0]])

    # assume the camera and world share the same coordinate system
    # [I|0]
    I = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    vec_0 = np.array([[0, 0, 0]]).T

    # camera matrix can be decomposed into two different matrices
    # P = K[I|0]
    P = K.dot(np.hstack((I, vec_0)))

    

camera_matrix()
    