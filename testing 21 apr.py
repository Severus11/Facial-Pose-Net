import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import Facial_Pose_Net_model

faceCas = cv2.CascadeClassifier('detector_architectures\haarcascade_frontalface_default.xml')
model = Facial_Pose_Net_model('model_new_latest.json', 'model_weights_LATEST.h5')


fr= cv2.imread('WIN_20210416_19_44_13_Pro.jpg')
sunglass = cv2.imread('sunglasses.png',cv2.IMREAD_UNCHANGED)

#gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
faces = faceCas.detectMultiScale(fr, 1.3, 5)
for (x, y, w, h) in faces:
    fc = fr[y:y+h, x:x+w]
    roi = cv2.resize(fc, (96,96))
    pred = model.predict_points(roi[np.newaxis, :, :])
    pred= pred.astype('uint8').reshape(-1,2)
    
    xs = int(pred[17, 0])
    ys = int(pred[17, 1])
    
    hs = abs(int(pred[27,1]) - int(pred[34,1]))
    ws = abs(int(pred[17,0]) - int(pred[26,0]))
    
    new_sunglasses= cv2.resize(sunglass, (ws,hs))
    ind = np.argwhere(new_sunglasses[:,:,3] > 0)
    
    for i in range(3):
        roi[ind[:,0],ind[:,1],i] = new_sunglasses[ind[:,0],ind[:,1],i]   
        
    plt.imshow(roi)


cv2.destroyAllWindows()    