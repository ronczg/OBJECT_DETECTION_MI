import cv2
import time
import os
import HandTrackingModule as htm

wCam,hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []
        #hüvelykujj
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        #többiujj
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        
        cv2.rectangle(img,(20,225),(170,425),(0,255,50),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),20)
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

cap.release()
cv2.destroyAllWindows()
   
