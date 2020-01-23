import cv2
import dlib
import numpy as np
# get a reference to our webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

# this function will take a rect from dlib and draw a box
def drawBox(dlibRect, frame):
        x = r.left()
        y = r.top()
        w = r.right() - r.left()
        h = r.bottom() - r.top()
        frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0),2)

while True:
    # read the stream
    # ret - true if success
    # frame - the video frame
    ret, frame = cap.read()
    # mirror the frame
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # for every face found draw a box around it
    for r in rects:
        drawBox(r, frame)
            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()