import numpy as np
import cv2 
from utils import img_path

path = r'D:\\Coding Project\\3d reconstruction\\images\\'

img1 = cv2.imread(img_path(path, 1), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_path(path, 2), cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)


bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
# apply ratio test
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

def find_features_pts(keypoints1, keypoints2):
    src_pts = np.asarray([keypoints1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.asarray([keypoints2[m.trainIdx].pt for m in good_matches])

    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    pts1 = src_pts[mask == 1]
    pts2 = src_pts[mask == 1]

    return pts1.T, pts2.T

pts1, pts2 = find_features_pts(keypoints1, keypoints2)
print(pts1.shape, pts2.shape)
print(pts1[0][:] == (pts2[0][:]))
print(pts1[1][:] == (pts2[1][:]))

matching_result = cv2.drawMatches(img1, 
                                          keypoints1, 
                                          img2, 
                                          keypoints2, 
                                          good_matches[0:500:20], 
                                          None, 
                                          flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Matching Result', matching_result)
cv2.waitKey(0)  
cv2.destroyAllWindows()