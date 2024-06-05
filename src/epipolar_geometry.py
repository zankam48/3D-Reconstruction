import numpy as np
import cv2
# from scipy import linalg

# pts1 = point in image1, pts2 = point in image2
from find_features import pts1, pts2 

# find fundamental matrix 8 points algorithm
def find_fundamental_matrix(pts1, pts2):
    '''
    xn'.T(F)xn = 0, where n = 1, ..., n
    0. (Normalize points)
    1. Construct the Mx9 matrix A
    2. Find the SVD of A.T*A
    3. Entries of F are the elements of column of V corresponding to the least singular value 
    4. (Enforce rank 2 constraint on F)
    5. (Un-normalize F)
    '''
    pts_idx = [0,1,2,3,4,5,6,7,8,9,10]
    f1,f2,f3,f4,f5,f6,f7,f8,f9 = int, int, int, int, int, int, int, int, int

    pts1_x, pts1_y = pts1[0], pts1[1]
    pts2_x, pts2_y = pts2[0], pts2[1]

    # homogenous linear system, AX = 0, if we choose 8 points hence pts_idex = 1, ..., 8
    # Total least squares -> minimize ||Ax||^2, subject to ||x||^2 = 1, USE SVD!

    # to save computation, i write the terms manually
    A = np.array([
            pts1_x[pts_idx]*pts2_x[pts_idx], pts1_x[pts_idx]*pts2_y[pts_idx], pts1_x,
            pts1_y[pts_idx]*pts2_x[pts_idx], pts1_y[pts_idx]*pts2_y[pts_idx], pts1_y,
            pts2_x[pts_idx], pts2_y[pts_idx], 1
        ])
    X = [f1, f2, f3, f4, f5, f6, f7, f8, f9]


    # epipole
    # Fe = 0, if the epipole in the right null space of F
    
    # with F known, then we can calculate K, R and t
    # F = K'^-T [tx] R K^-1

def find_fundamental_matrix_gem(pts1, pts2):
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    def normalized_pts(pts):
        x = pts[0]
        y = pts[1]
        centroid = np.mean(pts[:2], axis=1)
        cx = x - centroid[0]
        cy = y - centroid[1]
        dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
        scale = np.sqrt(2) / np.mean(dist)
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        pts_normalized = np.dot(T, pts)
        return T, pts_normalized
    
    T1, pts1_normalized = normalized_pts(pts1)
    T2, pts2_normalized = normalized_pts(pts2)

    A = np.zeros((8, 9))
    for i in range(8):
        x1, y1 = pts1_normalized[0][i], pts1_normalized[1][i]
        x2, y2 = pts2_normalized[0][i], pts2_normalized[1][i]
        A[i] = np.array([x2*x1, x1*y2, x1, y1*x2, y2*y1, y1, x2, y2, 1])
    
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2,2]


# gua
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

    # to align camera and world coordinate
    # Xc = R(Xw- C), genral mapping of a pinhole camera P = KR[I|-C] or P = K[R|t] where t = -RC

def pose_estimation():
    pass

def find_essential_matrix(pts1, pts2):
    pass

def intrinsic_matrix():
    pass


pts1_riil = np.array([
                        [100, 120, 130, 150, 180, 200, 220, 240, ],
                        [150, 170, 200, 250, 230, 260, 270, 300, ],
                        [1, 1, 1, 1, 1, 1, 1, 1, ]
                     ])

pts2_riil = np.array([
                        [102, 122, 132, 152, 182, 202, 222, 242, ],
                        [148, 168, 198, 248, 228, 258, 268, 298, ],
                        [1, 1, 1, 1, 1, 1, 1, 1, ]   
                    ])

F = find_fundamental_matrix_gem(pts1_riil, pts2_riil)
print(F)
print(np.sum(F))
