from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse

import numpy as np
import cv2


def select_white_yellow(image):
	converted = convert_hls(image)
	# white color mask
	lower = np.uint8([  0, 200,   0])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(converted, lower, upper)
	# yellow color mask
	lower = np.uint8([ 10,   0, 100])
	upper = np.uint8([ 40, 255, 255])
	yellow_mask = cv2.inRange(converted, lower, upper)
	# combine the mask
	mask = cv2.bitwise_or(white_mask, yellow_mask)
	return cv2.bitwise_and(image, image, mask = mask)

def convert_hsv(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def convert_gray_scale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_smoothing(image, kernel_size=15):
	"""
	kernel_size must be postivie and odd
	"""
	return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=20, high_threshold=60):
	return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
	"""
	Create the mask using the vertices and apply it to the input image
	"""
	mask = np.zeros_like(image)
	if len(mask.shape)==2:
		cv2.fillPoly(mask, vertices, 255)
	else:
		cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
	return cv2.bitwise_and(image, mask)

def select_region(image):
	"""
	It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
	"""
	# first, define the polygon by vertices
	rows, cols = image.shape[:2]
	bottom_left  = [cols*0, rows*0.90]
	top_left     = [cols*0.1, rows*0.65]
	bottom_right = [cols*1, rows*0.90]
	top_right    = [cols*0.9, rows*0.65]
	# the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	return filter_region(image, vertices)

def hough_lines(image):
	"""
	`image` should be the output of a Canny transform.
	Returns hough lines (not the image with lines)
	"""
	return cv2.HoughLinesP(image, rho=1, theta=np.pi/2, threshold=120, minLineLength=2, maxLineGap=100)
# model images
zebra = cv2.imread('/home/bobz/repos/OpenCV-programs/visionhack/zebra/test_images/Zebras/48.png',0)
zebra_by = cv2.imread('./test_images/Zebras/47.png',0)
zebra_corb = cv2.imread('./test_images/Zebras/74.png',0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, zbr = sift.detectAndCompute(zebra,None)
kp2, zbr_by = sift.detectAndCompute(zebra_by,None)
kp3, zbr_corb = sift.detectAndCompute(zebra_corb,None)

def SIFT(img2):
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(zbr,des2,k=2)
	matches_by = flann.knnMatch(zbr_by,des2,k=2)
	matches_corb = flann.knnMatch(zbr_corb,des2,k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.425*n.distance:
			good.append([m,n])
	size = len(good)

	good_by = []
	for m,n in matches_by:
		if m.distance < 0.425*n.distance:
			good_by.append([m,n])
	size_by = len(good_by)

	good_corb = []
	for m,n in matches_corb:
		if m.distance < 0.425*n.distance:
			good_corb.append([m,n])
	size_corb = len(good_corb)

	return max(size, size_by, size_corb)

def find_zebra(video_file):
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	cv2.moveWindow("frame", 0,0);
	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600,600)
	cv2.moveWindow("mask", 620,0);

	lines_prev = deque(maxlen=6)
	frame_num = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		# frame2 = frame
		# frame2 = convert_hls(frame)
		# frame2 = select_white_yellow(frame2)
		grey = convert_gray_scale(frame)
		equ = cv2.equalizeHist(grey)
		blurred = apply_smoothing(equ)
		# canned = detect_edges(blurred)
		region = select_region(blurred)
		# try:
		# 	print SIFT(region);
		# except:
		# 	pass

		cv2.imshow('frame',frame)
		cv2.imshow('mask',region)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(u'\u0020'):
			cv2.waitKey(-1)
		elif key == ord('q'):
			break
		frame_num = frame_num + 1
	cap.release()

if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'

	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	for file_name in file_list:
	  	find_zebra(path + file_name)
	# find_zebra(path + 'akn.281.131.left.avi')
