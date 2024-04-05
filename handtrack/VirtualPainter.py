import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "OBJECT_DETECTION_MI\handtrack\header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        print(lmList)
        
        
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        
    
    img[0:125,0:1280] = header
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)