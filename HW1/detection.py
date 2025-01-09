import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped

def acc(a,b):
    l = len(a)    
    count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            count += 1
    return float(count/l)
  
def detect(data_path, clf):
    width = 36
    height = 16
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    boxes = []
    f = open(data_path)
    chops = int(f.readline())
    for i in range(chops):
        l = f.readline()
        boxes.append(tuple(l.split(' ')))
    # Begin your code (Part 4)
    video = cv2.VideoCapture('data/detect/video.gif')
  
    f.close()
    f = open('ML_Models_pred.txt', mode = 'w')

    i = 0
    
    occupy = []
    while True:
        if i == 50:
            break
        i += 1
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        
        k = True
        c = 0
        for j in boxes:
            if k:
                k = False
            else:
              f.write(' ')  
            img = crop(j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7], frame)
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = clf.classify([img.flatten()])
            if result == 1:
                f.write('1')
                c += 1
                if i == 1:
                    
                    pts = np.array([ [j[2], j[3]],  [j[6], j[7]], [j[4], j[5]], [j[0], j[1]]], np.int32)

                    
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                f.write('0')
          
        f.write('\n')
        if i == 1:
            # cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        occupy.append(c)
        
    f.close()
    return occupy
    # End your code (Part 4)
