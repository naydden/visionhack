from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

import cv2
import numpy as np

def select_blue(image):
	converted = convert_hsv(image)
	# white color mask
	lower = np.uint8([100, 0,   0])
	upper = np.uint8([130, 255, 255])
	blue_mask = cv2.inRange(converted, lower, upper)
	# combine the mask
	return cv2.bitwise_and(image, image, mask = blue_mask)
def convert_hsv(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# model images
pass_es = cv2.imread('./test_images/pass_es_g.png',0)
pass_dr = cv2.imread('./test_images/pass_dr_g.png',0)
pass_lev = cv2.imread('./test_images/level_g.png',0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des_es = sift.detectAndCompute(pass_es,None)
kp1, des_dr = sift.detectAndCompute(pass_dr,None)
kp1, des_lev = sift.detectAndCompute(pass_lev,None)

def ORB(img2):
	rst, dest2 = cv2.threshold(img2, 130, 255, cv2.THRESH_BINARY)

	# find the keypoints and descriptors with SIFT
	kp2, des2 = orb.detectAndCompute(dest2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	print matches[0].distance
	# Draw first 10 matches.
	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
	# plt.imshow(img3),plt.show()

def SIFT(img2):
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches_es = flann.knnMatch(des_es,des2,k=2)
	matches_dr = flann.knnMatch(des_dr,des2,k=2)
	matches_lev = flann.knnMatch(des_lev,des2,k=2)
	# store all the good matches as per Lowe's ratio test.
	good_es = []
	for m,n in matches_es:
		if m.distance < 0.425*n.distance:
			good_es.append([m,n])
	size_es = len(good_es)

	good_dr = []
	for m,n in matches_dr:
		if m.distance < 0.425*n.distance:
			good_dr.append([m,n])
	size_dr = len(good_dr)

	good_lev = []
	for m,n in matches_lev:
		if m.distance < 0.425*n.distance:
			good_lev.append([m,n])
	size_lev = len(good_lev)

	if(size_es > size_lev):
		return (size_es,'z') if size_es > size_dr else (size_dr,'z')
	elif(size_dr > size_lev):
		return (size_dr, 'z') if size_dr > size_es else (size_es,'z')
	elif(size_es == size_lev):
		return (size_es, 'z') if good_es[0].distance < good_lev[0].distance else (size_lev,'l')
	elif(size_dr == size_lev):
		return (size_dr, 'z') if good_dr[0].distance < good_lev[0].distance else (size_lev, 'l')
	else:
		return (size_lev, 'l')

zebra = False
level = False
def defineTrafficSign(image):
	global zebra
	global level
	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	image = imutils.resize(image, height=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 50, 200, 255)
	# find contours in the edge map, then sort them by their
	# size in descending order
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	# The is_cv2() and is_cv3() are simple functions that can be used to
	# automatically determine the OpenCV version of the current environment
	# cnts[0] or cnts[1] hold contours
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.13 * peri, True)

		# if the contour has four vertices, then we have found
		# the thermostat display
		if len(approx) == 4:
			displayCnt = approx
			break

	# extract the sign borders, apply a perspective transform
	# to it
	# A common task in computer vision and image processing is to perform
	# a 4-point perspective transform of a ROI in an image and obtain a top-down, "birds eye view" of the ROI
	if (displayCnt is not None):
		warped = four_point_transform(gray, displayCnt.reshape(4, 2))
		output = four_point_transform(image, displayCnt.reshape(4, 2))
		# draw a red square on the image
		cv2.drawContours(image, [displayCnt], -1, (0, 0, 255), 5)
		blue_mask = select_blue(output)
		gray_mask = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
		equalized = cv2.equalizeHist(gray_mask)
		nbr = cv2.countNonZero(equalized)
		if (nbr > 200):
			try:
				size, of = SIFT(output);
				if(int(size) > 1):
					if (of == 'z'):
						zebra = True;
					elif (of == 'l'):
						level = True
			except:
				pass
			# cv2.imshow('mask',output)
	# extract_triangle(output)

	# threshold the warped image, then apply a series of morphological
	# operations to cleanup the thresholded image
	# cv2.THRESH_OTSU. it automatically calculates a threshold value from image histogram
	# for a bimodal image
	# thresh = cv2.threshold(warped, 0, 255,
	# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def find(video):
	global zebra
	global level
	zebra = False
	level = False
	# cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('frame', 600,600)
	# cv2.moveWindow("frame", 0,0);
	# cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('mask', 600,600)
	# cv2.moveWindow("mask", 620,0);
	cap = cv2.VideoCapture(video)
	while(cap.isOpened()):
		(grabbed, frame) = cap.read()
		if grabbed == False:
			break
		# cv2.imshow('frame',frame)
		defineTrafficSign(frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord(u'\u0020'):
			cv2.waitKey(-1)
		elif key == ord('q'):
			cv2.destroyAllWindows()
			print("Stop programm and close all windows")
			break
	cap.release()
def bool2num(bool):
	return 1 if bool else 0
if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/AI/Hack/trainset/'
	file = open('./test/results-valid', 'w')
	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	print len(file_list)
	for indx, file_name in enumerate(file_list):
		find(path + file_name)
		query_level = str(bool2num(level))
		query_zebra = str(bool2num(zebra))
		save = file_name +" "+query_level+query_zebra+"\n"
		file.write(save)
		print "-----------------------------" + str(indx) + " "+file_name
	file.close()

