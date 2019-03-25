e# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
import random as rn
from tqdm import tqdm
import os   
import cv2    

from keras import models
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


model = models.load_model('/home/mitchell/Desktop/4301Project/WepDet.h5')

X=[]
Z=[]
IMG_SIZE=150
DATA_GUN_DIR='/home/mitchell/Desktop/4301Project/dataset/test_set/gun'
DATA_NONGUN_DIR='/home/mitchell/Desktop/4301Project/dataset/test_set/nogun'

def assign_label(img,img_type):
    return img_type

def make_test_data(img_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        #print(img)
        if not img.startswith('.'):
            label=assign_label(img,img_type)
            path = os.path.join(DIR,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
            Z.append(str(label))
        
make_test_data('Gun',DATA_GUN_DIR)
print(len(X))        

make_test_data('nogun',DATA_NONGUN_DIR)
print(len(X))

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        # Had index oob error here once
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Gun: '+Z[l])
        
plt.tight_layout()

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break


