from os import listdir
from os.path import isfile, join
from collections import deque
from matplotlib import pyplot as plt
from operator import sub

import math
import argparse

import time

import numpy as np
import cv2

def find_movement(img1,img2):
    # Initiate SIFT detector

    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

#    print kp1
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

    aux=10
    if aux>len(matches):
        aux=len(matches)

    #print " auux: " + str(aux) + "---  leen: " + str((len(matches)))
    mitjana=0

    if matches is None:
        print "a la mierdaaaa"
#
    for x in range (0,aux):
        mitjana+=(kp2[x].pt[1]-kp1[x].pt[1])/aux

    return mitjana

def detect_movement(text_file, video_file):
    cap = cv2.VideoCapture(video_file)

    txt = mypath + file_name[:11] + (file_name[11:] and '.txt')

    file_handle = open(txt, 'r')
    lines_list = file_handle.readlines()
    x_coord, y_coord = (int(val) for val in lines_list[0].split())
    file_handle.close()

    x=0
    d=0
    aux=0

#    distances = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            break

        crop_img = frame[y_coord-150:y_coord+150, x_coord-100:x_coord+100] # Crop from x, y, w, h -> 100, 200, 300, 400

        if x!=0:
            i=find_movement(img,crop_img)
            if x!=1:
#                distances.append(i-aux)
                aux2=abs(i-aux)
                if aux2>=95:
                    d=1
            aux=i
        x+=1

        img=crop_img

#        cv2.imshow("cropped", crop_img)
#        cv2.waitKey(0)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#    plt.plot(distances) # plotting by columns
#    plt.show()
    text_file.write(file_name+" "+str(d)+"\n")

    cap.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=False, help="path to videos directory")
    args = vars(ap.parse_args())

    mypath = 'C:/Users/crist/Desktop/VisionHack/trainset/frames/'

    if args['folder'] is not None:
        mypath = args['folder']
    print(mypath)
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[-1] == 'avi']
    text_file = open("Output.txt", "a")
    for file_name in file_list:
        detect_movement(text_file, mypath + file_name)


    text_file.close()
