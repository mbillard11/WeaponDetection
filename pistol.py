
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:56:20 2019

@author: mitchell
"""

import cv2
import numpy as np
from PIL import Image
from keras import models
import matplotlib.pyplot as plt

#Load the saved model
model = models.load_model('/home/mitchell/Desktop/4301Project/WepDet.h5')
video = cv2.VideoCapture('/home/mitchell/Desktop/4301Project/pistol.mp4')

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((150,150))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array)[0][0]
        print(prediction)

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()