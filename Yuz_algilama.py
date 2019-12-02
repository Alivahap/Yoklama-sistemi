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

    yuzler = yuz.detectMultiScale(gri_renk,1.5,5)


    for(x,y,w,h) in yuzler:
      cv2.rectangle(kare,(x,y),(x+w,y+h),(0,0,255),2)

      roi_color = kare[y:y + h, x:x + w]
      roi_gray = gri_renk[y:y + h,x:x+w]
      id_, dogruluk = recognizer.predict(roi_gray)




      if id_ == 0:
          print("ali Vahap")
          isim = "Ali Vahap"
      elif id_ == 1:
          print("Eren")
          isim = "Eren"
      elif id_ == 2:
          print("Onur")
          isim = "Onur"

      cv2.putText(kare, isim, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

      if dogruluk >= 65:
          print(id_)

















    cv2.imshow("Video", kare)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

kmr.release()
cv2.destroyAllWindows()

