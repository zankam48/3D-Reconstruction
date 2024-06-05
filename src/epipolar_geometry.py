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
        centroid = np.mean(pts, axis=1)
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
        A[i] = np.array([x2*x1, x1*y2, x1, y1*x2, y2*y1, y1, x2, y2, 1]).T
    
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F

# alyssa reconstruction

def correspondence_matrix(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([
        p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y,
        p2x, p2y, np.ones(len(p1x))
    ]).T


def compute_image_to_image_matrix(x1, x2, compute_essential=False):
    """ Compute the fundamental or essential matrix from corresponding points
        (x1, x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    A = correspondence_matrix(x1, x2)
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def scale_and_translate_points(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: array of homogenous point (3 x n)
    :returns: array of same input shape and its normalization matrix
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    norm3d = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])

    return np.dot(norm3d, points), norm3d


def compute_normalized_image_to_image_matrix(p1, p2, compute_essential=False):
    """ Computes the fundamental or essential matrix from corresponding points
        using the normalized 8 point algorithm.
    :input p1, p2: corresponding points with shape 3 x n
    :returns: fundamental or essential matrix with shape 3 x 3
    """
    n = p1.shape[1]
    if p2.shape[1] != n:
        raise ValueError('Number of points do not match.')

    # preprocess image coordinates
    p1n, T1 = scale_and_translate_points(p1)
    p2n, T2 = scale_and_translate_points(p2)

    # compute F or E with the coordinates
    F = compute_image_to_image_matrix(p1n, p2n, compute_essential)

    # reverse preprocessing of coordinates
    # We know that P1' E P2 = 0
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]


def compute_fundamental_normalized(p1, p2):
    return compute_normalized_image_to_image_matrix(p1, p2)


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

pts1 = np.array([
    [100, 150],
    [120, 170],
    [130, 200],
    [150, 250],
    [180, 230],
    [200, 260],
    [220, 270],
    [240, 300]
])

pts1_riil = np.array([
                        [100, 120, 130, 150, 180, 200, 220, 240, 260, 280],
                        [150, 170, 200, 250, 230, 260, 270, 300, 340, 360],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                     ])

pts2_riil = np.array([
                        [102, 122, 132, 152, 182, 202, 222, 242, 252, 267],
                        [148, 168, 198, 248, 228, 258, 268, 298, 318, 325],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   
                    ])

pts2 = np.array([
    [102, 148],
    [122, 168],
    [132, 198],
    [152, 248],
    [182, 228],
    [202, 258],
    [222, 268],
    [242, 298]
])

F = find_fundamental_matrix_gem(pts1_riil, pts2_riil)
print(F)
print(np.sum(F))

Fa = compute_fundamental_normalized(pts1_riil, pts2_riil)
print(Fa)
print(np.sum(Fa))