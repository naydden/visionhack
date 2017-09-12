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
        vrnc = np.array([])
        frm_nmbr = 0
        while(cap.isOpened()):
            retval, frame = cap.read()
            if retval == True:
                frm_nmbr += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # edges = cv2.Canny(gray,50,150,apertureSize = 3)
                # lines = cv2.HoughLines(edges,1,np.pi/180,200)
                # for line in lines[0:1]:
                #     for rho,theta in line:
                #         a = np.cos(theta)
                #         b = np.sin(theta)
                #         x0 = a*rho
                #         y0 = b*rho
                #         x1 = int(x0 + 1000*(-b))
                #         y1 = int(y0 + 1000*(a))
                #         x2 = int(x0 - 1000*(-b))
                #         y2 = int(y0 - 1000*(a))
                #         cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
                # frames = np.append(frames, frm_nmbr)
                if frm_nmbr % 12 == 0:
                    mean = np.mean(gray)
                    var = np.var(gray)
                    brghtns = np.append(brghtns, mean)
                    vrnc = np.append(vrnc, var)
                cv2.imshow('res',gray)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                break
        # gradient = np.gradient(brghtns)
        # avg = np.average(brghtns)
        # brgmax = np.amax(brghtns)
        # varmax = np.amax(vrnc)
        # quot1 = brghtns/brgmax
        # quot2 = vrnc/varmax
        # P = 1.0 - quot
        # diff = np.amax(gradient) - np.amin(gradient)
        diff = np.amax(brghtns) - np.amin(brghtns)
        if diff > 48:
            print f + ' ' + '100000' + ' ' + str(diff)
            fid.write(f + ' ' + '100000' + '\n')
        else:
            print f + ' ' + '000000' + ' ' + str(diff)
            fid.write(f + ' ' + '000000' + '\n')
        # plt.plot(brghtns)
        # plt.plot(quot1)
        # plt.plot(quot2)
        # plt.plot(P)
        # plt.plot([0, 300], [avg, avg])
        # plt.show()
        # plt.savefig(os.path.join(path, 'figures', f + '.png'))
        # plt.clf()
        cv2.waitKey(0)
