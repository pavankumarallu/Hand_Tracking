import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False,1,0.6,0.6)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0




while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if  results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for ids,lm in enumerate(handLMS.landmark):
                
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                if ids == 8:
                    cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)
            
            
            mpDraw.draw_landmarks(img,handLMS,mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(int(fps)),(20,130),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    
    cv2.imshow("Image",img)
    
    
    
    
    
    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()