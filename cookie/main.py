#INITIAL SETUP
#----------------------------------------------------------------
import cv2
from cvzone import HandTrackingModule, overlayPNG
import numpy as np
intro = cv2.imread("frames/img1.jpeg", cv2.IMREAD_COLOR)
kill = cv2.imread("frames/img2.png", cv2.IMREAD_COLOR)
winner = cv2.imread("frames/img3.png", cv2.IMREAD_COLOR)
cam = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector(maxHands=1,detectionCon=0.77)
#sets the minimum confidence threshold for the detection

#INITILIZING GAME COMPONENTS
#----------------------------------------------------------------
sqr_img = cv2.imread("img/sqr(2).png", cv2.IMREAD_COLOR)
mlsa =  cv2.imread("img/mlsa.png", cv2.IMREAD_COLOR)
#INTRO SCREEN WILL STAY UNTIL Q IS PRESSED

while True:
    cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.69, fy=0.69))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

gameOver = False
NotWon =True
#GAME LOGIC UPTO THE TEAMS
#-----------------------------------------------------------------------------------------
while not gameOver:
        alpha = 0.4
        isTrue, frame = cam.read()
        foreground = cv2.resize(sqr_img, (600, 600), fx= 0.69, fy = 0.69)
        background = cv2.resize(frame, (600, 600), fx= 0.69, fy = 0.69)
        hands, img = detector.findHands(background, flipType=True)

        
        

        added_image = cv2.addWeighted(background,1,foreground,1-alpha,0)

        cv2.imshow('Squid Game', cv2.resize(added_image, (900, 675), fx = 0.69, fy = 0.69))



     

        if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                centerPoint1 = hand1['center']  # center of the hand cx,cy
                handType1 = hand1["type"]  # Handtype Left or Right

                fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)
        # cv2.imshow('Video', frame)


        if(cv2.waitKey(20) & 0xFF==ord('q')):
                gameOver = True
                break

if NotWon:
    for i in range(10):
        cv2.imshow('LOSS', cv2.resize(kill, (0, 0), fx=0.69, fy=0.69))
          
    while True:
        cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.69, fy=0.69))                
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

frame.release()
cv2.destroyAllWindows()