import cv2

# get a reference to our webcam
cap = cv2.VideoCapture(0)

# Loop until we stop with a 'q' keystoke
while True:
    # read the stream
    # ret - true if success
    # frame - the video frame
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
