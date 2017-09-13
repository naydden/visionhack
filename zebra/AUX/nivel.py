from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import glob
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
kp, des_z = sift.detectAndCompute(np.array([pass_es,pass_dr]),None)
kp1, des_lev = sift.detectAndCompute(pass_lev,None)

images = []
for file in glob.glob("/home/bobz/repos/OpenCV-programs/visionhack/zebra/test_images/bad/*.png"):
	img = cv2.imread(file,0)
	images.append(img)
kps, des_bad = sift.detectAndCompute(np.array(images),None)

def SIFT(img2):
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches_z = flann.knnMatch(des_z,des2,k=2)

	matches_lev = flann.knnMatch(des_lev,des2,k=2)

	matches_bad = flann.knnMatch(des_bad,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good_z = []
	for m,n in matches_z:
		if m.distance < 0.45*n.distance:
			good_z.append([m,n])
	size_z = len(good_z)
	good_lev = []
	for m,n in matches_lev:
		if m.distance < 0.45*n.distance:
			good_lev.append([m,n])
	size_lev = len(good_lev)

	good_bad = []
	for m,n in matches_bad:
		if m.distance < 0.45*n.distance:
			good_bad.append([m,n])
	size_bad = len(good_bad)

	if(size_z > size_bad):
		return (size_z,'z') if size_z > size_lev else (size_lev,'l')
	elif(size_lev > size_bad):
		return (size_lev, 'l') if size_lev > size_z else (size_z,'z')
	elif(size_z == size_bad):
		return (size_z, 'z') if good_z[0].distance < good_bad[0].distance else (0,'b')
	elif(size_lev == size_bad):
		return (size_lev, 'l') if good_lev[0].distance < good_bad[0].distance else (0, 'b')
	else:
		return (0, 'b')

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
		if (nbr > 300):
			try:
				size,of = SIFT(output)
				print size
				if(int(size) > 1):
					if (of == 'z'):
						zebra = True
						cv2.waitKey(-1)
					elif (of == 'l'):
						level = True
						cv2.waitKey(-1)
			except:
				pass
			cv2.imshow('mask',output)

def find(video):
	global zebra
	global level
	zebra = False
	level = False
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	cv2.moveWindow("frame", 0,0);
	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600,600)
	cv2.moveWindow("mask", 620,0);
	cap = cv2.VideoCapture(video)
	while(cap.isOpened()):
		(grabbed, frame) = cap.read()
		if grabbed == False:
			break
		cv2.imshow('frame',frame)
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
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'
	# file = open('./test/results-valid', 'w')
	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	print len(file_list)
	for indx, file_name in enumerate(file_list):
		# if(file_name == "akn.289.048.left.avi"):
		find(path + file_name)
		query_level = str(bool2num(level))
		query_zebra = str(bool2num(zebra))
		save = file_name +" "+query_level+query_zebra+"\n"
		# file.write(save)
		print "-----------------------------" + str(indx) + " "+file_name
		print save
	# file.close()

