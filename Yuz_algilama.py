import cv2
import numpy as np

kmr = cv2.VideoCapture(0)

while True:
    ret,kare = kmr.read()
    yuz = cv2.CascadeClassifier('haarcascade-frontalface-default.xml ')
   # gulme=cv2.CascadeClassifier('haarcascade_smile.xml ')

    gri_renk = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
  #  gri_renk_gulme=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)

    faces = yuz.detectMultiScale(gri_renk,1.3,4)

    for(x,y,w,h) in faces:
      cv2.rectangle(kare,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Video", kare)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

kmr.release()
cv2.destroyAllWindows()

