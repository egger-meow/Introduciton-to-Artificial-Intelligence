import cv2
import numpy as np

import random



filePath = 'data/image.png'

img = cv2.imread(filePath)
height, width, channels = img.shape

img1 = img / 255


img2 = img[::-1]


a1 = random.randint(0,int(height*0.6))
b1 = random.randint(0,int(width*0.6))

a2 = random.randint(int(a1+height*0.3),int(height))
b2 = random.randint(int(b1+width*0.3),int(width))

img3 = img[a1:a2,b1:b2]


rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 37, 1)

img4 = cv2.warpAffine(img, rotation_matrix, (width, height))


brightness_factor = 1.6

img5 = img * brightness_factor if  (img * brightness_factor).any() <255 else 255

     
sharpen = np.array(
    [
        [-1,-1,-1],
        [-1,10,-1],
        [-1,-1,-1]
    ]
)

img6 = cv2.filter2D(img,-1,sharpen) 

cv2.imwrite('img1.png',img1)
cv2.imwrite('img2.png',img2)
cv2.imwrite('img3.png',img3)
cv2.imwrite('img4.png',img4)
cv2.imwrite('img5.png',img5)
cv2.imwrite('img6.png',img6)

