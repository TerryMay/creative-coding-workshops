import cv2
import dlib
import numpy as np
from imutils import face_utils

# some properties
show_face_rect = False
show_landmarks = True 
show_pose_line = True
# last translation vector
last_tvec = np.zeros((3,1), dtype='float32')
# last rotation vector
last_rvec = np.zeros((3,1), dtype='float32')
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
# landmark_indicies = [36, 39, 42, 45, 30]
landmark_indicies = [29]
color_map = []
for idx in range(70):
    color_map.append((225, 225, 0) if idx in landmark_indicies else (0, 0, 255))

# this defines points in 3D space of what we expect a human face to look like
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (-48.825, 36.89, -29.295),  # Left eye left corner
    (-15.825, 36.89, -29.647), # left eye right corner
    (48.825, 36.89, -29.647),  # Right eye right corne
    (15.825, 36.89, -29.295)
])

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

def shape_to_image_points(shape):
    return np.array([
        shape[29], #nose tip
        shape[36], #left eye left corner
        shape[39], #left eye right corner
        shape[45], #right eye right corner
        shape[42], #right eye left corner
    ], dtype="double")

while True:
    # read the stream
    # ret - true if success
    # frame - the video frame
    ret, frame = cap.read()
    # mirror the frame
    frame = cv2.flip(frame,1)
    if size == None:
        size = frame.shape
        print("[INFO] window size = " + str(size))

        # camera intrinsics
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    for face in rects:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        image_points = shape_to_image_points(shape)

        (success, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, \
            dist_coeffs, useExtrinsicGuess=True, tvec=last_tvec, rvec=last_rvec, flags=cv2.SOLVEPNP_ITERATIVE)
        
        last_tvec = tvec
        last_rvec = rvec

        if (show_face_rect):
            draw_box(frame, face)
        if (show_landmarks):
            draw_landmarks(frame, shape)    
        if show_pose_line:
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            lr = int(shape[45][0]), int(shape[45][1])
            rl = int(shape[39][0]), int(shape[39][1])
            mid = int((rl[0] + lr[0])/2), int((rl[1] + lr[1])/2)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rvec,
                tvec, camera_matrix, dist_coeffs)

            p1 = (int(shape[29][0]), int(shape[29][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(frame, mid, p2, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()