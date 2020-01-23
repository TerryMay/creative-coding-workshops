import cv2

# get a reference to our webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# get the frontal face HAAR Cascade
face_cascade = cv2.CascadeClassifier('../../support/classifiers/haarcascade_frontalface_default.xml')

# Loop until we stop with a 'q' keystoke
while True:
    # read the stream
    # ret - true if success
    # frame - the video frame
    ret, frame = cap.read()

    # process the frame here
    
    # convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use the classifier to find a face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for every face found draw a box around it
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
    #

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
