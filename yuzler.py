import os
from PIL import Image
import numpy as np
import cv2
import pickle


yuz = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')


#temel klasör dosyalarımızın temelde nerede olduğunun bilgisini verir
temel_klasor = os.path.dirname(os.path.abspath(__file__))
resim_klasor=os.path.join(temel_klasor,"resimler")
#resim klasör resimlerimizin nerede olduğunun bilgisini döndürür

#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

#recognizer = cv2.createLBPHFaceRecognizer()
#recognizer=cv2.face.EigenFaceRecognizer_create()
#x= cv2.createLBHFaceRecognizer()
y_labels=[]
x_train=[]
current_id=0
label_ids={}

for kok_dizin,dizin,dosyalar in os.walk(resim_klasor):
 for dosya in dosyalar:

     if dosya.endswith("png") or dosya.endswith("jpg") or dosya.endswith("jpeg"):

         path = os.path.join(kok_dizin,dosya)
         label=os.path.basename(os.path.dirname(path).replace(" ","-")).lower()
         #print(label,path)
      #   y_labels.append(label)
       #  x_train.append(path)
         if not label in label_ids:
             label_ids[label]=current_id
             current_id+=1

         id = label_ids[label]
         print(label_ids)


         pil_resim=Image.open(path).convert("L")
         resim_array = np.array(pil_resim, "uint8")

         yuzler = yuz.detectMultiScale(resim_array, 1.5, 5)
         for(x,y,w,h) in yuzler:
             roi=resim_array[y:y+h,x:x+w]
             x_train.append(roi)
             y_labels.append(id)

with open("etiket.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")








