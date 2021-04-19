import cv2
import numpy as np
from model import Facial_Pose_Net_model

faceCas = cv2.CascadeClassifier('detector_architectures\haarcascade_frontalface_default.xml')
model = Facial_Pose_Net_model('model_new_latest.json', 'model_weights_LATEST.h5')

cap = cv2.VideoCapture(0)

while True:
    _, fr = cap.read()
    #gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCas.detectMultiScale(fr, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = fr[y:y+h, x:x+w]
        roi = cv2.resize(fc, (96,96))
        pred = model.predict_points(roi[np.newaxis, :, :])
        pred= pred.astype('uint8').reshape(-1,2)
        for i in range(15,60):
            cv2.circle(roi,(pred[i,0], pred[i,1]), 2, (255,0,0), 7)
    
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('FKPD',roi)
    
    if cv2.waitKey(50) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()  