import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

cwd = os.getcwd()
path = os.path.join(cwd, 'data', 'trainset', 'bridges')
# path = os.path.join(cwd, 'data', 'trainset')
# path = os.path.join(cwd, 'data', 'validationset')

fid = open(os.path.join(path, 'maxmin_dif.txt'), 'w+')

for f in os.listdir(path):
    if f[-4:] == '.avi':
        cap = cv2.VideoCapture(os.path.join(path, f))
        # frames = np.array([])
        brghtns = np.array([])
        frm_nmbr = 0
        while(cap.isOpened()):
            retval, frame = cap.read()
            if retval == True:
                frm_nmbr += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray)
                # frames = np.append(frames, frm_nmbr)
                # if frm_nmbr % 12 == 0:
                brghtns = np.append(brghtns, mean)
                # cv2.imshow('res',gray)
                # k = cv2.waitKey(1) & 0xFF
                # if k == 27:
                #     break
            else:
                break
        gradient = np.gradient(brghtns)
        avg = np.average(brghtns)
        diff = np.amax(gradient) - np.amin(gradient)
        # diff = np.amax(brghtns) - np.amin(brghtns)
        if diff > 48:
            print f + ' ' + '100000' + ' ' + str(diff)
            fid.write(f + ' ' + '100000' + '\n')
        else:
            print f + ' ' + '000000' + ' ' + str(diff)
            fid.write(f + ' ' + '000000' + '\n')
        plt.plot(brghtns)
        plt.plot(gradient)
        plt.plot([0, 300], [avg, avg])
        # plt.show()
        plt.savefig(os.path.join(path, 'figures', f + '.png'))
        plt.clf()
        # cv2.waitKey(0)
