from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse

import numpy as np
import cv2

# define range of white color in HSV
# change it according to your need !
lower_white = np.array(150, dtype=np.uint8)
upper_white = np.array(255, dtype=np.uint8)

def find_zebra(video_file):
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	cv2.moveWindow("frame", 0,0);
	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600,600)
	cv2.moveWindow("mask", 620,0);

	while(cap.isOpened()):
		ret, frame = cap.read()

		if frame is None:
			break
		frame_h,frame_w, _ = frame.shape
		crop_h, crop_w = int(0.6 * frame_h), int(0.5 * frame_w)

		# frame_crop = frame[crop_h:(frame_h-200),0:frame_w]
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(hsv, (7, 7), 0.5)
		# Threshold the HSV image to get only white colors
		mask = cv2.inRange(blurred, lower_white, upper_white)
		# Bitwise-AND mask and original image
		# res = cv2.bitwise_and(frame,frame, mask= mask)
		mask = cv2.Canny(mask, 1, 3)
		im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnt = contours[0]
		# x,y,w,h = cv2.boundingRect(cnt)
		# hull = cv2.convexHull(cnt)
		# cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),2)
		epsilon = 0.1*cv2.arcLength(cnt,True)
		approx = cv2.approxPolyDP(cnt,epsilon,True)
		x,y,w,h = cv2.boundingRect(approx)
		cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,12),4)
		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)

		key = cv2.waitKey(1) & 0xFF
		if key == ord(u'\u0020'):
			cv2.waitKey(-1)
		elif key == ord('q'):
			break

		# cv2.waitKey(1000/2)
	cap.release()

if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'

	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	for file_name in file_list:
		print file_name
		find_zebra(path + file_name)
