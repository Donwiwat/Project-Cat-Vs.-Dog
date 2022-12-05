import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        handsType = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)
                
            
            for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx = int(lm.x*w)
                    cy = int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
                    if id == 4:
                        cv2.rectangle(img, pt1=(cx-120,cy-120), pt2=(cx+120,cy+120), color=(255,0,0), thickness=10)
        return lmList, handsType

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = detector.findHands(img)
        lmList, handsType = detector.findPosition(img)
        print(handsType)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
    
    
