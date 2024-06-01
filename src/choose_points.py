import cv2
import numpy as np
from find_features import img1, img2, pts1, pts2
from math import sqrt

class ChooseFeatures:
    def __init__(self, img1=None, img2=None):
        self.img1 = img1
        self.img2 = img2
        self.src_pts = []
        self.dst_pts = []
    
    def select_pts(self, event, x, y, flags, params):
        global src_x, src_y
        if event == cv2.EVENT_LBUTTONDBLCLK:
            src_x, src_y = x, y

            distances = []
            for x_features, y_features in zip(pts1[0], pts1[1]):
                x_nearest = x_features - x
                y_nearest = y_features - y
                distance = sqrt(x_nearest**2 + y_nearest**2)
                distances.append(distance)
    
            nearest_index = np.argmin(distances)
            nearest_x_img1 = pts1[0][nearest_index]
            nearest_y_img1 = pts1[1][nearest_index]

            nearest_x_img2 = pts2[0][nearest_index]
            nearest_y_img2 = pts2[1][nearest_index]

            cv2.circle(self.img1, (int(nearest_x_img1), int(nearest_y_img1)), 5, (255, 255, 255), -1)
            cv2.circle(self.img2, (int(nearest_x_img2), int(nearest_y_img2)), 5, (255, 255, 255), -1)

            self.src_pts.append([nearest_x_img1, nearest_y_img1])
            self.dst_pts.append([nearest_x_img2, nearest_y_img2])
            print(self.src_pts, self.dst_pts)


    def random_select_pts(self, n_pts: int, dist_pts: str):
        pass

cv2.namedWindow('image_1')

features = ChooseFeatures(img1=img1, img2=img2)

cv2.setMouseCallback('image_1', features.select_pts)

while True:
    cv2.imshow('image_1', img1)
    cv2.imshow('image_2', img2)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

