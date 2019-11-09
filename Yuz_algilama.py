import os
from PIL import Image
import numpy as np
import cv2
import pickle

kmr = cv2.VideoCapture(0)
yuz = cv2.CascadeClassifier('haarcascade-frontalface-default.xml ')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

while True:
    ret,kare = kmr.read()



   # gulme=cv2.CascadeClassifier('haarcascade_smile.xml ')

    gri_renk = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
  #  gri_renk_gulme=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)

    yuzler = yuz.detectMultiScale(gri_renk,1.3,4)

    for(x,y,w,h) in yuzler:
      cv2.rectangle(kare,(x,y),(x+w,y+h),(0,0,255),2)
      roi_color = kare[y:y + h, x:x + w]
      roi_gray = gri_renk[y:y + h,x:x+w]
      id_, dogruluk = recognizer.predict(roi_gray)
      if dogruluk >=40 and dogruluk<=85:
          print(id_)

    if id_ == 0:
        print("ali vahap")
    else:
        print("Halit Ergenç")



    cv2.imshow("Video", kare)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

kmr.release()
cv2.destroyAllWindows()

