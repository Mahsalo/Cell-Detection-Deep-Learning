#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:50:30 2018
@author: Mahsa
"""
from skimage import io
import imageio
import os
import glob
import keras
import scipy
import skimage
import numpy as np
from os import listdir
from PIL import Image as PImage
import cv2
import math
import pandas as pd
from keras import regularizers
import matplotlib.pyplot as plt
np.random.seed(2016)
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Input, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import model_from_json
from skimage.transform import rotate
from keras.models import load_model 
from keras.utils import to_categorical


train_X=[]
train_temp=[]
test_X=[]
row_tr=[]
col_tr=[]  
row_ts=[]
col_ts=[]
target=[]
target_temp=[]
r_tr=[]
c_tr=[]

#####Function for finding the one's complement of a binary 2D array
def flipp(x):
    m=x.shape[0]###rows
    n=x.shape[1]###columns
    for i in range(m):
        for j in range(n):
            if x[i][j]==255:
                x[i][j]=1
            else:
                x[i][j]=0
    return x            

####binarization of Target_cells using Otsu
filenames3=[imm for imm in glob.glob("./target/*.jpg")]
for image_path in filenames3:
    imgg = cv2.imread(image_path,0)
    ret,th = cv2.threshold(imgg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ###data augmentation to overcome overfitting
    rott1=rotate(th,30).astype('uint8')
    rott2=rotate(th,45).astype('uint8')
    rott3=rotate(th,60).astype('uint8')
    rott4=rotate(th,90).astype('uint8')
    rott5=rotate(th,180).astype('uint8')
    rott6=rotate(th,270).astype('uint8')
       
    rott7=rotate(th,20).astype('uint8')
    rott8=rotate(th,50).astype('uint8')
    rott9=rotate(th,70).astype('uint8')
    rott10=rotate(th,90).astype('uint8')
    rott11=rotate(th,290).astype('uint8')
    rott12=rotate(th,125).astype('uint8')
    rott13=rotate(th,310).astype('uint8')
    th=flipp(th)
    target.append(th)
    target_temp.extend([rott1,rott2,rott3,rott4,rott5,rott6,rott7,rott8,rott9,rott10,rott11,rott12,rott13])
#target.extend(target_temp)
#target_len=len(target)

####target test files
filenames4=[imm1 for imm1 in glob.glob("./target_test/*.jpg")]
target_test=[]   
for image_path in filenames4:
    imgg4 = cv2.imread(image_path,0)
    ret1,th1 = cv2.threshold(imgg4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th1=flipp(th1)
    target_test.append(th1)

####Reading the training images 
filenames = [img for img in glob.glob("./training/*.jpg")]
#for image_path in glob.glob("./training/*.jpg"):
for image_path in filenames:
    im = cv2.imread(image_path,0)
    #plt.imshow(im,cmap='gray')
    ###data augmentation to overcome overfitting
    rot1=rotate(im,30).astype('uint8')
    rot2=rotate(im,45).astype('uint8')
    rot3=rotate(im,60).astype('uint8')
    rot4=rotate(im,90).astype('uint8')
    rot5=rotate(im,180).astype('uint8')
    rot6=rotate(im,270).astype('uint8')

    train_X.append(im)
    train_temp.extend([rot1,rot2,rot3,rot4,rot5,rot6])
    row_tr.extend([im.shape[0], rot1.shape[0], rot2.shape[0],rot3.shape[0],rot4.shape[0],rot5.shape[0],rot6.shape[0]])
    col_tr.extend([im.shape[1], rot1.shape[1], rot2.shape[1],rot3.shape[1],rot4.shape[1],rot5.shape[1],rot6.shape[1]])
    #row_tr.append(im.shape[0])
    #col_tr.append(im.shape[1])
 
max_row=row_tr[np.argmax(row_tr)]
max_col=col_tr[np.argmax(col_tr)]
#max_row=254
#max_col=254

first_len=len((train_X))  
sec_len=len((train_temp))

for i in range(len(train_X)):### Zeropadding to reach the maximum rows and columns with value of 255 
    rows=train_X[i].shape[0]
    cols=train_X[i].shape[1]
    x_dif=max_row-rows
    y_dif=max_col-cols
    constant= cv2.copyMakeBorder(train_X[i],x_dif,0,y_dif,0,cv2.BORDER_CONSTANT,value=255)
    train_X[i]=constant
    [l1,l2]=np.where(train_X[i]==255)###this gives the values as a list in l1 and l2
    [l3,l4]=np.where(train_X[i]!=255)
    img_tr=train_X[i]
    #print(train_X[i].shape)
    l1=list(l1)
    l2=list(l2)
    l3=list(l3)
    l4=list(l4)
    for j in range(len(l1)):
        img_tr[l1[j],l2[j]]=0
    for j in range(len(l3)):    
        img_tr[l3[j],l4[j]]=1
    train_X[i]=img_tr  
      
for i in range(len(train_temp)):### Zeropadding to reach the maximum rows and columns with value of 255 
    rows1=train_temp[i].shape[0]
    cols1=train_temp[i].shape[1]
    x_dif1=max_row-rows1
    y_dif1=max_col-cols1
    constant1= cv2.copyMakeBorder(train_temp[i],x_dif1,0,y_dif1,0,cv2.BORDER_CONSTANT,value=0)
    train_temp[i]=constant1
    [l1,l2]=np.where(train_temp[i]!=0)###this gives the values as a list in l1 and l2
    img_tr_temp=train_temp[i]
    #print(train_X[i].shape)
    l1=list(l1)
    l2=list(l2)

    for j in range(len(l1)):
        img_tr_temp[l1[j],l2[j]]=1
    train_temp[i]=img_tr_temp 
train_X.extend(train_temp)  

#####change the size of the target cells
for i in range(len(target)):### Zeropadding to reach the maximum rows and columns with value of 255 
    rows2=target[i].shape[0]
    cols2=target[i].shape[1]
    x_dif2=max_row-rows2
    y_dif2=max_col-cols2
    constant2= cv2.copyMakeBorder(target[i],x_dif2,0,y_dif2,0,cv2.BORDER_CONSTANT,value=0)
    target[i]=constant2
    

train_X.extend(target) ####add all target cells afterwards to the training set

####Reading the test images

for i in range(len(target_test)):### Zeropadding to reach the maximum rows and columns with value of 255 
    rows3=target_test[i].shape[0]
    cols3=target_test[i].shape[1]
    x_dif3=max_row-rows3
    y_dif3=max_col-cols3
    constant3= cv2.copyMakeBorder(target_test[i],x_dif3,0,y_dif3,0,cv2.BORDER_CONSTANT,value=0)
    target_test[i]=constant3

filenamestest = [img for img in glob.glob("./test_set/*.jpg")]

for img in filenamestest:
    img1 = cv2.imread(img,0)
    test_X.append(img1)
    row_ts.append(img1.shape[0])
    col_ts.append(img1.shape[1])      

for i in range(len(test_X)):### Zeropadding to reach the maximum rows and columns with value of 255 
    rows4=test_X[i].shape[0]
    cols4=test_X[i].shape[1]
    x_dif4=max_row-rows4
    y_dif4=max_col-cols4
    constant4= cv2.copyMakeBorder(test_X[i],x_dif4,0,y_dif4,0,cv2.BORDER_CONSTANT,value=255)
    test_X[i]=constant4
    [l5,l6]=np.where(test_X[i]==255)###this gives the values as a list in l5 and l6
    [l7,l8]=np.where(test_X[i]!=255)
    img_test=test_X[i]
    #print(train_X[i].shape)
    l5=list(l5)
    l6=list(l6)
    l7=list(l7)
    l8=list(l8)
    for j in range(len(l5)):
        img_test[l5[j],l6[j]]=0
    for j in range(len(l7)):    
        img_test[l7[j],l8[j]]=1
    test_X[i]=img_test 
    
test_X.extend(target_test) 
   
temp_tr=train_X
temp_test=test_X    
train_X=np.array(train_X)
test_X=np.array(test_X)    
train_X=train_X.reshape(-1,max_row,max_col,1)
test_X=test_X.reshape(-1,max_row,max_col,1)
    
####Forming the one-hot vector labels for training and test sets
tr_y=np.zeros((len(train_X),11))###we have 11 different classes 
tr_y[0:98,0]=1;###class1: Hypochrom cells
tr_y[98:196,1]=1;###class2: Normal cells
tr_y[196:294,2]=1;### class3: Sickle cells
tr_y[294:392,3]=1;### class4: Schistocyte cells
tr_y[392:419,4]=1;### class5: Spurious
tr_y[419:513,5]=1;### class6: Elliptocytes
tr_y[513:595,6]=1;### class7: Knizocytes
tr_y[595:692,7]=1;### class8: Stomatocyte
tr_y[692:791,8]=1;### class9: Macrocyte
tr_y[791:889,9]=1;### class10: Tear Drop cell

tr_y[889:1477,0]=1;###class1: Hypochrom cells
tr_y[1477:2065,1]=1;###class2: Normal cells
tr_y[2065:2653,2]=1;### class3: Sickle cells
tr_y[2653:3241,3]=1;### class4: Schistocyte cells
tr_y[3241:3403,4]=1;### class5: Spurious
tr_y[3403:3967,5]=1;### class6: Elliptocytes
tr_y[3967:4459,6]=1;### class7: Knizocytes
tr_y[4459:5041,7]=1;### class8: Stomatocyte
tr_y[5041:5635,8]=1;### class9: Macrocyte
tr_y[5635:6223,9]=1;### class10: Tear Drop cell

tr_y[6223:len(train_X),10]=1;###class11: target cells

ts_y=np.zeros((len(test_X),11))
ts_y[0,0]=1
ts_y[1:4,1]=1
ts_y[4:8,4]=1
ts_y[8:11,2]=1
ts_y[11,0]=1
ts_y[12,2]=1
ts_y[13:17,3]=1
ts_y[17:21,9]=1
ts_y[21,5]=1
ts_y[22,0]=1
ts_y[23:28,5]=1
ts_y[28:32,6]=1
ts_y[32,7]=1
ts_y[33,0]=1
ts_y[34:44,7]=1
ts_y[44,0]=1
ts_y[45:47,8]=1
ts_y[47:50,9]=1
ts_y[50:53,0]=1
ts_y[53,1]=1
ts_y[54:len(test_X),10]=1###targets

####vector model of the one_hot_vector label

Ts_Y=np.zeros((len(test_X),1))
Ts_Y[[0,11,22,33,44,50,51,52],0]=0
Ts_Y[[1,2,3,53],0]=1
Ts_Y[[8,9,10,12],0]=2
Ts_Y[[13,14,15,16],0]=3
Ts_Y[[4,5,6,7],0]=4
Ts_Y[[21,23,24,25,26,27],0]=5
Ts_Y[[28,29,30,31],0]=6
Ts_Y[[32,34,35,36,37,38,39,40,41,42,43],0]=7
Ts_Y[[45,46],0]=8
Ts_Y[[17,18,19,20,47,48,49],0]=9
Ts_Y[54:len(test_X)]=10

###Make a validation set
tr_X, valid_X, tr_Y, valid_Y=train_test_split(train_X,tr_y,test_size=0.2, random_state=13)

####Let's define CLOT regularizer
#mu=0.99###the weight of the l1 and l2 in the regularizer
#clot_reg=keras.regularizers.l1_l2(mu,(1-mu))
#el2_reg=keras.regularizers.l2(0.01)
el1_reg=keras.regularizers.l2(0.01)

###Define the model
batch_size=20
epochs=50
num_classes=11
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(max_row, max_col,1), padding='same', kernel_regularizer=el1_reg))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(64,kernel_size=(3,3), activation='relu', padding='same',kernel_regularizer=el1_reg))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=(3,3), activation='relu', padding='same',kernel_regularizer=el1_reg))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.22))
model.add(Conv2D(256,kernel_size=(3,3), activation='relu', padding='same',kernel_regularizer=el1_reg))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(512,kernel_size=(3,3), activation='relu', padding='same',kernel_regularizer=el1_reg))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


###compiling and training
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
training_step=model.fit(tr_X, tr_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_Y))
test_eval=model.evaluate(test_X, ts_y, verbose=0)
####prediction
Predicted=model.predict(test_X)
Predicted_round=np.argmax(np.round(Predicted),axis=1)
Predicted_real=Predicted_round.reshape(len(test_X),1)
dif=Predicted_real-Ts_Y
Incorrect=np.where(dif!=0)
num_inc=len(Incorrect[0])
test_acc=100-num_inc*100/len(test_X)

###let's draw overfitting plots

#accuracy=training_step.history['acc']
#valid_acc=training_step.history['val_acc']
#loss=training_step.history['loss']
#valid_loss=training_step.history['val_loss']
#epochs=range(len(accuracy))
#plt.plot(epochs,accuracy,'bo',label='Training accuracy')
#plt.plot(epochs,valid_acc,'b',label='Validation accuracy')
#plt.title('Training Accuracy Vs. Validation Accuracy Using l2 Regularizer')
#plt.legend()
#plt.show()
#plt.figure()%%%open up another figure

###loading
##from keras.models import load_model
##new_model=load_model('./Desktop/workspace_with_targets.h5py')

###saving the model and all variables
###keep all the variables inside a list then pass the list to the pickle
### import pickle
### with open('.....pickle','wb') as f:
### pickle.dump(list,f)
####list_total=[Incorrect,Predicted,Predicted_real,Predicted_round,Ts_Y,batch_size,c_tr,col_tr,col_ts,cols,cols1,cols2,cols3,cols4,dif, epochs,max_col,max_row,num_classes,row_tr,row_ts,target,target_temp,target_test, temp_test,temp_tr,test_X,test_acc,test_eval,tr_X,tr_Y,train_X,train_temp,valid_X,valid_Y,ts_y]

##Plotting
beingsaved = plt.figure()
##l1
a=[1.64,1.4606,1.3169,1.3473,1.2465,1.2013,1.1984,1.1712,1.2006,1.2010,1.1658,1.1509,1.1227,1.1591,1.2525,1.224,1.1428,1.1792,1.2846,1.2449,1.1270,1.2051,1.2087,1.1865,1.2972,1.1987,1.1820,1.2129,1.2278,1.1906,1.1448,1.1848,1.1861,1.1632,1.2016,1.1484,1.152,1.1289,1.1322,1.098,1.15,1.1542,1.1564,1.1031,1.0544,1.0991,1.1161,1.0903,1.1157,1.1206]
##l2
b=[1.6252,1.437,1.3213,1.3529,1.2574,1.1989,1.2013,1.185,1.2045,1.2248,1.1772,1.2668,1.1683,1.1729,1.1979,1.2798,1.1781,1.2046,1.3079,1.2806,1.1753,1.2137,1.2094,1.2354,1.3270,1.2531,1.2011,1.3278,1.2351,1.1982,1.1705,1.2021,1.2540,1.1556,1.1453,1.2238,1.238,1.1905,1.2118,1.0836,1.1822,1.1365,1.1072,1.1397,1.087,1.0737,1.1489,1.1453,1.1073,1.1247]
##Dropout
c=[0.9393,0.8727,0.6109,0.5672,0.6154,0.6099,0.604,0.7184,0.979,0.8423,0.7636,0.6714,0.7336,0.8453,0.8493,0.8126,0.8509,0.798,0.9182,0.8615,0.8846,0.7541,0.8953,0.8247,0.9845,0.7534,0.8408,0.8481,0.8301,1.0628,0.8696,1.0087,1.0216,0.7668,0.9394,1.0908,0.6759,1.0388,0.8673,0.8099,0.9081,1.0793,1.0696,1.0711,1.0804,1.2647,1.2113,1.0406,1.1087,1.0664]
##both
d=[1.9812,1.6301,1.4311,1.4033,1.345,1.3449,1.2917,1.3077,1.3149,1.2654,1.2699,1.2801,1.284,1.2378,1.3169,1.1858,1.2749,1.2147,1.2257,1.1997,1.2705,1.1993,1.2485,1.3591,1.225,1.2881,1.2316,1.2878,1.2306,1.2436,1.2587,1.3029,1.2813,1.2449,1.2784,1.2673,1.2801,1.2713,1.2551,1.265,1.2439,1.1602,1.2819,1.2026,1.2637,1.1875,1.2336,1.2239,1.3042,1.3058]
z=range(1,51)
fig=plt.plot(z,a,label="L1 Regularizer")
fig=plt.plot(z,b,label="L2 Regularizer")
fig=plt.plot(z,c,label="Dropout Layer")
fig=plt.plot(z,d,label="L1 + Dropout")
plt.xlabel('Epochs')
plt.ylabel('Validation Cross Entropy Loss')
plt.title('Loss vs. Epochs for one fixed validation set')
plt.legend()
#fig.savefig('myimage.jpg', format='jpg', dpi=1200)
beingsaved.savefig('myimage.eps', format='eps', dpi=1200)
