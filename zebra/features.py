import numpy as np
import cv2
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

def detect_edges(image, low_threshold=20, high_threshold=60):
	return cv2.Canny(image, low_threshold, high_threshold)

def ORB(img1,img2):
	# Initiate SIFT detector
	orb = cv2.ORB_create()
	rst, dest1 = cv2.threshold(img1, 130, 255, cv2.THRESH_BINARY)
	rst, dest2 = cv2.threshold(img2, 130, 255, cv2.THRESH_BINARY)

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	for m in matches:
		print m.distance
	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)
	plt.imshow(img3),plt.show()

def SIFT(img1,img2):
	MIN_MATCH_COUNT = 10
	rst, dest1 = cv2.threshold(img1, 130, 255, cv2.THRESH_BINARY)
	rst, dest2 = cv2.threshold(img2, 130, 255, cv2.THRESH_BINARY)
	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)
	print matches
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.55*n.distance:
			good.append([m,n])
	# Sort them in the order of their distance.
	good = sorted(good, key = lambda x: x[0].distance)
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:10],None,flags=2)

	plt.imshow(img3),plt.show()


img1 = cv2.imread('./test_images/b.png',0) # queryImage
img2 = cv2.imread('./test_images/a.png',0) # trainImage
ORB(img1,img2)
