from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse
from matplotlib import pyplot as plt

import numpy as np
import cv2


def select_white_yellow(image):
	converted = convert_hls(image)
	# white color mask
	lower = np.uint8([  0, 100,   0])
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

def select_region(image, coor_x, coor_y):
	if (coor_y > 600):
		coor_y = 500
	"""
	It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
	"""
	# first, define the polygon by vertices
	if (image is not None):
		rows, cols = image.shape[:2]
		bottom_left  = [cols*0.3, rows*0.9]
		top_left     = [cols*0.4, rows*0.7]
		bottom_right = [cols*0.7, rows*0.9]
		top_right    = [cols*0.6, rows*0.7]
		# the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
		vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
		return filter_region(image, vertices)
	else:
		return image

def hough_lines(image):
	"""
	`image` should be the output of a Canny transform.
	Returns hough lines (not the image with lines)
	"""
	return cv2.HoughLinesP(image, rho=1.7, theta=0.1, threshold=160, minLineLength=50, maxLineGap=100)

def find_zebra(video_file, focus):
	data = ''
	with open(focus, 'r') as myfile:
		data = myfile.read().replace('\n', '')
	data = data.split()
	data = map(int, data)
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	cv2.moveWindow("frame", 0,0);
	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600,600)
	cv2.moveWindow("mask", 620,0);
	n_black = np.array([])
	lines_prev = deque(maxlen=6)
	frame_num = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break;
		hls = convert_hls(frame)
		frame2 = select_white_yellow(hls)
		# region = select_region(frame, data[0], data[1])
		# grey = convert_gray_scale(region)
		# blurred = apply_smoothing(grey)
		# canned = detect_edges(blurred)
		# nbr = cv2.countNonZero(grey)
		# n_black = np.append(n_black, nbr)
		cv2.imshow('frame',frame)
		cv2.imshow('mask',frame2)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(u'\u0020'):
			cv2.waitKey(-1)
		elif key == ord('q'):
			break
		frame_num = frame_num + 1
	cap.release()
	# plt.plot(n_black)
	# plt.show()
	# plt.clf()

if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'

	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	focus_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'txt']
	for indx, file_name in enumerate(file_list):
	  	find_zebra(path + file_name, path + focus_list[indx])
	# find_zebra(path + 'akn.211.002.left.avi', path + 'akn.211.002.txt')
