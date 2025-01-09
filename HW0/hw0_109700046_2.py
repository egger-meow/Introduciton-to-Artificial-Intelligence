import cv2
import numpy as np

# Path to the MP4 file
file_path = 'data/video.mp4'

# Open the MP4 file
cap = cv2.VideoCapture(file_path)

ret, frame = cap.read()

height, width, channels = frame.shape

h = int(height / 5)
w = int(width / 5)

frameFilter = cv2.resize(frame,(w,h))
    
dif0 = np.zeros(frameFilter.shape, dtype=frame.dtype)

print(dif0.shape)
# frameFilter = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


c = 0

while True:
    c += 1
    
    ret, frame = cap.read()
 
        
    frame = cv2.resize(frame,(w,h))
    
    # f = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    dif = cv2.absdiff(frame, frameFilter)
    

    difdif = cv2.absdiff(dif, dif0)

    img = np.hstack((frame, difdif))
    if c == 15:
        cv2.imwrite('test.png',img)
    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frameFilter = frame
    dif0 = dif


cap.release()
cv2.destroyAllWindows()