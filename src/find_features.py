import numpy as np
import cv2 
from utils import img_path

path = r'D:\\Coding Project\\3d reconstruction\\images\\'

img1 = cv2.imread(img_path(path, 1), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_path(path, 2), cv2.IMREAD_GRAYSCALE)


class FindFeatures:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

        # sift detectors
        sift = cv2.SIFT_create()
        self.keypoints1, self.descriptors1 = sift.detectAndCompute(img1, None)
        self.keypoints2, self.descriptors2 = sift.detectAndCompute(img2, None)

        self.good_matches = []

    def find_matches(self):
        # bruteforce matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.descriptors1, self.descriptors2, k=2)

        # apply ratio test
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                self.good_matches.append(m)
        return self.good_matches

    def find_features_pts(self):
        good_matches = self.find_matches()
        src_pts = np.asarray([self.keypoints1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.asarray([self.keypoints2[m.trainIdx].pt for m in good_matches])

        retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        mask = mask.ravel()

        pts1 = src_pts[mask == 1]
        pts2 = dst_pts[mask == 1]

        return pts1.T, pts2.T

    # optional function to check
    def draw_matches(self):
        self.find_matches()
        matching_result = cv2.drawMatches(img1, 
                                          self.keypoints1, 
                                          img2, 
                                          self.keypoints2, 
                                          self.good_matches, 
                                          None, 
                                          flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        return matching_result

features = FindFeatures(img1, img2)
pts1, pts2 = features.find_features_pts()
print(pts1.shape, pts2.shape)
print(pts1[0][:] == (pts2[0][:]))
print(pts1[1][:] == (pts2[1][:]))

# showing the result
# cv2.imshow('Feature Matching Result', features.draw_matches())
# cv2.waitKey(0)  
# cv2.destroyAllWindows()
