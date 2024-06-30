import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

width, height = 1280, 720
folderPath = "presentation"

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# list of slides
pathImages = sorted(os.listdir(folderPath), key=len)
# print(pathImages)

imgNum = 0
hs, ws = 120, 213
gestureThreshold = 300
buttonPress = False
buttonCounter = 0
buttonDelay = 25
annotations = [[]]
annotationNumber = -1
annotationStart = False

# hand detector
detector = HandDetector(detectionCon=0.8)

while True:
    # import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNum])
    imgCurr = cv2.imread(pathFullImage)

    # Check if the image was successfully read
    if imgCurr is None:
        print(f"Error: Image at {pathFullImage} could not be loaded.")
        continue

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 7)

    if hands and buttonPress is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # constrain value for easy drawing
        indexFinger = lmList[8][0], lmList[8][1]

        # Interpolate the coordinates to fit the slide dimensions
        xVal = int(np.interp(lmList[8][0], [0, width], [0, imgCurr.shape[1]]))
        yVal = int(np.interp(lmList[8][1], [0, height], [0, imgCurr.shape[0]]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # if hand at height of face
            annotationStart = False
            # gesture 1 left
            if fingers == [1, 0, 0, 0, 0]:
                print("left")
                if imgNum > 0:
                    buttonPress = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNum -= 1

            # gesture 2 right
            if fingers == [0, 0, 0, 0, 1]:
                print("right")
                if imgNum < len(pathImages) - 1:
                    buttonPress = True
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    imgNum += 1

        # gesture 3 pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        # gesture 4 draw pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurr, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False
        
        # gesture 5
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPress = True
    else:
        annotationStart = False

    # button press iteration
    if buttonPress:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPress = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurr, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurr.shape
    imgCurr[0:hs, w - ws:w] = imgSmall
    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurr)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
