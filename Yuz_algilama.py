import cv2
import numpy as np

kmr = cv2.VideoCapture(0)

while (kmr.isOpened()):
    ret,kare = kmr.read()

    yuz = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gri = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    yuzler = yuz.detectMultiScale(gri,1.1,4)
    for(x,y,w,h) in yuzler:
      cv2.rectangle(kare,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Video", kare)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

kmr.release()
cv2.destroyAllWindows()

