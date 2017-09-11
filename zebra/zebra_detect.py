from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse

import numpy as np
import cv2

def find_zebra(video_file):
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 600,600)
	cv2.moveWindow("frame", 0,0);
	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600,600)
	cv2.moveWindow("mask", 620,0);
	# cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('res', 600,600)
	while(cap.isOpened()):
		ret, frame = cap.read()

		if frame is None:
			break

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# define range of white color in HSV
		# change it according to your need !
		lower_white = np.array([0,0,0], dtype=np.uint8)
		upper_white = np.array([0,0,255], dtype=np.uint8)

		# Threshold the HSV image to get only white colors
		mask = cv2.inRange(hsv, lower_white, upper_white)
		# Bitwise-AND mask and original image
		res = cv2.bitwise_and(frame,frame, mask= mask)
		cv2.imshow('frame',frame)
		cv2.imshow('mask',mask)
		# cv2.imshow('res',res)

		# cv2.imshow("images", frame_crop)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		# cv2.waitKey(1000/2)
	cap.release()

if __name__ == "__main__":
	path = '/home/bobz/repos/OpenCV-programs/visionhack/zebra/data/'

	file_list = [f for f in listdir(path) if isfile(join(path, f)) and f.split('.')[-1] == 'avi']
	for file_name in file_list:
		print file_name
		find_zebra(path + file_name)
