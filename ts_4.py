import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

cap = cv.VideoCapture(0)
cas = cv.CascadeClassifier()
if not (cas.load("haarcascade_frontalface_alt.xml")):
    print ("load xml file failed!")
    exit(-1)

if not (cap.isOpened()):
    print ("cap open failed!")
    cap.release()
    exit(-1)

index = 1
while(1):
    ret,img = cap.read()
    if not (ret == False):
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        faces = cas.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            pt1 = (x,y)
            pt2 = (x+w,y+h)
            cv.rectangle(img,pt1,pt2,(255,255,0))
            tmpsout = img[y:(y+h),x:(x+w)]
            dirs = "../data/xny/face_"+str(index)+".jpg"
            # tmpsout = cv.cvtColor(tmpsout,cv.COLOR_RGB2GRAY)
            tmpsout = cv.resize(tmpsout,(320,320),interpolation=cv.INTER_AREA)
            cv.imwrite(dirs,tmpsout)
            index += 1
        cv.imshow("cap",img)
    if (index == 100):
        break
    key = cv.waitKey(30)
cap.release()
exit(0)
