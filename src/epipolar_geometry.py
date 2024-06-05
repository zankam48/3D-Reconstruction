import numpy as np
import cv2
# from scipy import linalg

# pts1 = point in image1, pts2 = point in image2
from find_features import pts1, pts2 

class F_E_Matrix():
    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2

    def normalized_pts(self, pts):
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
    
    def F_E_matrix(self, essential: bool):
        pts1 = np.asarray(self.pts1)
        pts2 = np.asarray(self.pts2)

        T1, pts1_normalized = self.normalized_pts(pts1)
        T2, pts2_normalized = self.normalized_pts(pts2)

        A = np.zeros((8, 9))
        for i in range(8):
            x1, y1 = pts1_normalized[0][i], pts1_normalized[1][i]
            x2, y2 = pts2_normalized[0][i], pts2_normalized[1][i]
            A[i] = np.array([x2*x1, x1*y2, x1, y1*x2, y2*y1, y1, x2, y2, 1])
        
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3,3)

        U, S, V = np.linalg.svd(F)
        if essential == False:
            S[-1] = 0
        else:
            S = [1, 1, 0]
        F = np.dot(U, np.dot(np.diag(S), V))
        F = np.dot(T1.T, np.dot(F, T2))

        return F / F[2,2]

    def find_fundamental_matrix(self):
        return self.F_E_matrix(essential=False)
    
    def find_essential_matrix(self):
        return self.F_E_matrix(essential=True)



def find_fundamental_matrix(pts1, pts2):
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

F = find_fundamental_matrix(pts1_riil, pts2_riil)
print(F)
print(np.sum(F))


find_F_E = F_E_Matrix(pts1_riil, pts2_riil)
Fa = find_F_E.find_fundamental_matrix()
Ea = find_F_E.find_essential_matrix()
# print(Fa)
# print(Ea)
# print(np.sum(Fa))
# print(np.sum(Ea))