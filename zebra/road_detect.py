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
	upper = np.uint8([140, 255, 255])
	blue_mask = cv2.inRange(converted, lower, upper)
	# combine the mask
	return cv2.bitwise_and(image, image, mask = blue_mask)
def convert_hsv(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def extract_triangle(image):
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
			approx = cv2.approxPolyDP(c, 0.1 * peri, True)

			# if the contour has four vertices, then we have found
			# the thermostat display
			if len(approx) == 3:
					displayCnt = approx
					break

	cv2.imshow('mask',displayCnt)

def defineTrafficSign(image):

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
				approx = cv2.approxPolyDP(c, 0.12 * peri, True)

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
			nbr = cv2.countNonZero(gray_mask)
			if (nbr > 200):
				cv2.imshow('mask',output)
		# extract_triangle(output)

		# threshold the warped image, then apply a series of morphological
		# operations to cleanup the thresholded image
		# cv2.THRESH_OTSU. it automatically calculates a threshold value from image histogram
		# for a bimodal image
		# thresh = cv2.threshold(warped, 0, 255,
		# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
		# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

		# # (roiH, roiW) = roi.shape
		# #subHeight = thresh.shape[0]/10
		# #subWidth = thresh.shape[1]/10
		# (subHeight, subWidth) = np.divide(thresh.shape, 10)
		# subHeight = int(subHeight)
		# subWidth = int(subWidth)
		# # mark the ROIs borders on the image
		# cv2.rectangle(output, (subWidth, 4*subHeight), (3*subWidth, 9*subHeight), (0,255,0),2) # left block
		# cv2.rectangle(output, (4*subWidth, 4*subHeight), (6*subWidth, 9*subHeight), (0,255,0),2) # center block
		# cv2.rectangle(output, (7*subWidth, 4*subHeight), (9*subWidth, 9*subHeight), (0,255,0),2) # right block
		# cv2.rectangle(output, (3*subWidth, 2*subHeight), (7*subWidth, 4*subHeight), (0,255,0),2) # top block
		# # # substract 4 ROI of the sign thresh image
		# leftBlock = thresh[4*subHeight:9*subHeight, subWidth:3*subWidth]
		# centerBlock = thresh[4*subHeight:9*subHeight, 4*subWidth:6*subWidth]
		# rightBlock = thresh[4*subHeight:9*subHeight, 7*subWidth:9*subWidth]
		# topBlock = thresh[2*subHeight:4*subHeight, 3*subWidth:7*subWidth]

		# # we now track the fraction of each ROI. (sum of active pixels)/(total number of pixels)
		# leftFraction = np.sum(leftBlock)/(leftBlock.shape[0]*leftBlock.shape[1])
		# centerFraction = np.sum(centerBlock)/(centerBlock.shape[0]*centerBlock.shape[1])
		# rightFraction = np.sum(rightBlock)/(rightBlock.shape[0]*rightBlock.shape[1])
		# topFraction = np.sum(topBlock)/(topBlock.shape[0]*topBlock.shape[1])

		# segments = (leftFraction, centerFraction, rightFraction, topFraction)
		# segments = tuple(1 if segment > 230 else 0 for segment in segments)



def find(video, txt):
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

if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'

	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	focus_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'txt']
	for indx, file_name in enumerate(file_list):
		find(path + file_name, path + focus_list[indx])
