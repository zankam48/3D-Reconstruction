import cv2
import numpy as np
from find_features import img1, img2, pts1, pts2

class ChooseFeatures:
    def __init__(self, img1=None, img2=None):
        self.img1 = img1
        self.img2 = img2
        self.src_pts = []
        self.dst_pts = []
    
    def select_src_pts(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img1, (x, y), 5, (255, 255, 255), -1)

    def select_dst_pts(self):
        pass

cv2.namedWindow('image')
features = ChooseFeatures(img1=img1)
cv2.setMouseCallback('image', features.select_src_pts)

while True:
    cv2.imshow('image', img1)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

