import cv2
import dlib
import numpy as np
from imutils import face_utils

# some properties
show_face_rect = False
show_landmarks = True 

# location of the shape predictor
shape_predictor = "../../support/shape_predictor_68_face_landmarks.dat"

# get a reference to our webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
size = None

# define a table of colors for landmarks of interest
landmark_indicies = [36, 39, 42, 45, 30]
color_map = []
for idx in range(70):
    color_map.append((225, 225, 0) if idx in landmark_indicies else (0, 0, 255))

# this function will take a rect from dlib and draw a box
def draw_box(frame, dlib_rect):
        x = r.left()
        y = r.top()
        w = r.right() - r.left()
        h = r.bottom() - r.top()
        frame = cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0),2)

def draw_landmarks(frame, detected_shape):
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for index, (x, y) in enumerate(detected_shape):
        cv2.circle(frame, (x, y), 1, color_map[index], -1)
        #cv2.putText(frame, "{}".format(index), (x, y), self.font, 1,(255, 0, 200),2, cv2.LINE_AA)

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

    for face in rects:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        if (show_face_rect):
            draw_box(frame, face)
        if (show_landmarks):
            draw_landmarks(frame, shape)    
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()