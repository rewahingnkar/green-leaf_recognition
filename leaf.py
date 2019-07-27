import cv2
import numpy as np

leaf_cascade=cv2.CascadeClassifier('D:\pycharm projects\leaf_cascade.XML')

img=cv2.imread('D:\pycharm projects\leaf_1.jpg',cv2.IMREAD_UNCHANGED)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

leaves= leaf_cascade.detectMultiScale(gray,1.3,5)
for(x,y,w,h) in leaves:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
