from PIL import Image
import imutils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dropout, BatchNormalization, Flatten, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import cv2
from tensorflow.keras.callbacks import Callback
#from keras import backend as K
from pathlib import Path
import sys
import numpy as np
from matplotlib.pyplot import imshow, figure
from tensorflow.keras.models import Model
import os
from tensorflow.keras.callbacks import History 
from pre_process import Preprocessed_additional_data
from pre_process import Preprocessed_1day_data


#Data preparation for training on additional data
path = "/home/prasad/SIH/oct/tir/"  
path = "/home/prasad/SIH/oct/vis/
Oct = os.listdir( path )
Oct=sorted(Oct)

path2 = "/home/prasad/SIH/nov_2019_data/tir/"
path2 = "/home/prasad/SIH/nov_2019_data/vis/"
Nov = os.listdir( path2 )
Nov=sorted(Nov)

Oct = [ elem for elem in Oct if int(elem[18:20]) % 30 == 0] 
Nov     = [ elem for elem in Nov if int(elem[18:20]) % 30 == 0] 
Nov=np.array(Nov)
Oct= np.array(Oct)

End_index=np.where(Nov=='3DIMG_06NOV2019_2300_L1C_SGP_IMG_TIR1.tif')

X,t,d,m,imgs_path=Load_additional_data("/home/prasad/SIH/oct/vis/",Oct[500], Oct[-1])
X1,t1,d1,m1,imgs_path1=Load_additional_data("/home/prasad/SIH/nov_2019_data/vis/",start_img_path=Nov[0],end_img_path=Nov[End_index[0][0]]) #
X2=np.vstack((X, X1))
t2=np.hstack((t, t1))
d2=np.hstack((d, d1))
m2=np.hstack((m, m1))
imgs_path2=np.hstack((imgs_path,imgs_path1))

X_train,y_train,X_val,y_val,maxx,minn,verify=Preprocessed_additional_data(X2,t2,d2,m2,imgs_path2,normalization_type ='scaling',inp_seq_len=3,pred_frame=1,normalized_output='no',Validation_split=0.2)


# Model architecture 
height, width=X_train.shape[2], X_train.shape[3]  
model = keras.Sequential(
    [
        keras.Input(
            shape=(None, height, width, 1)
        ),  # Variable-length sequence of 40x40x1 frames
        layers.ConvLSTM2D(
            filters=40, kernel_size=(5,5), padding="same", return_sequences=False
        ),       
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=10, kernel_size=(5,5), padding="same"
        ),       
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=5, kernel_size=(5,5), padding="same"
        ),       
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=1, kernel_size=(3,3), padding="same"
        ),
    ]
)

print(model.summary())
#print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(
                          monitor='loss',
                          min_delta=0,
                          patience=4,
                          verbose=1,restore_best_weights=True)

callbacks = [earlystop]
model.fit(X_train, y_train ,batch_size=1,epochs=150,callbacks=callbacks,validation_data=(X_val, y_val))
model.save('additionaldata_tir_model.h5')




## tuning on given sih data
from tensorflow import keras
model = keras.models.load_model('additionaldata_tir_model.h5' )

#data preparation
dir_path ="./../pytorch-unet/INSAT3D_VIS_India/" 
vis_list=os.listdir(dir_path)
vis_list=sorted(vis_list)

dir_path ="./../pytorch-unet/INSAT3D_TIR1_India/" 
tir_list=os.listdir(dir_path)
tir_list= sorted(tir_list)

tir_reg_list = [ elem for elem in tir_list if int(elem[18:20]) % 30 == 0] 
vis_reg_list     = [ elem for elem in vis_list if int(elem[18:20]) % 30 == 0] 

dir_path ="./../pytorch-unet/INSAT3D_VIS_India/" 
vis=[]
vis_name=[]
for i in range(len(vis_reg_list[0:25])):
    img = np.array(Image.open(os.path.join(dir_path,vis_reg_list[i]) ))
    vis.append(img)
    vis_name.append(vis_reg_list[i])
    
    
dir_path ="./../pytorch-unet/INSAT3D_TIR1_India/" 
tir=[]
tir_name=[]
for i in range(len(tir_reg_list)):
    img = np.array(Image.open(os.path.join(dir_path,tir_reg_list[i]) ))
    tir.append(img)
    tir_name.append(tir_reg_list[i])
    
tir=np.array(tir)

mean, std_dev = 654.0341745330628, 104.64211547894843   #tir.mean(),tir.std()
normalised_tir=[]
for img in tir:
    norm = ((img - mean)/std_dev) 
    normalised_tir.append(norm)
    
tir_1      = normalised_tir[:34]
raw_tir_1  = tir[:34]

tir_name_1 = tir_name[:34]

tir_2      = normalised_tir[34:]
raw_tir_2  = tir[34:]

tir_name_2 = tir_name[34:]

window_size=3  #10
output='no'

tir_x=[]
tir_y=[]
for i in range(len(tir_1)):
    if (i+window_size)>(len(tir_1)-1):
        break
    x=tir_1[i:i+window_size]
    if output=='norm':
        y=tir_1[i+window_size:i+window_size+1]
    else:
        y=raw_tir_1[i+window_size:i+window_size+1]
        
    tir_x.append(x)
    tir_y.append(y)

for i in range(len(tir_2)):
    if (i+window_size)>(len(tir_2)-1):
        break
    x=tir_2[i:i+window_size]
    if output=='norm':
        y=tir_2[i+window_size:i+window_size+1]
    else:
        y=raw_tir_2[i+window_size:i+window_size+1]
    tir_x.append(x)
    tir_y.append(y)

tir_x=np.array(tir_x)
tir_y=np.array(tir_y)

tir_x=np.reshape(tir_x, (tir_x.shape[0],tir_x.shape[1],tir_x.shape[2],tir_x.shape[3],1))  
tir_y=np.reshape(tir_y, (tir_y.shape[0],tir_y.shape[2],tir_y.shape[3],1))  


X= tir_x
y=tir_y

print('X.shape and y.shape:',X.shape, y.shape)
window_size=3  #10
output='no'

tir_x=[]
tir_y=[]
for i in range(len(tir_1)):
    if (i+window_size)>(len(tir_1)-1):
        break
    x=tir_1[i:i+window_size]
    if output=='norm':
        y=tir_1[i+window_size:i+window_size+1]
    else:
        y=raw_tir_1[i+window_size:i+window_size+1]
        
    tir_x.append(x)
    tir_y.append(y)


for i in range(len(tir_2)):
    if (i+window_size)>(len(tir_2)-1):
        break
    x=tir_2[i:i+window_size]
    if output=='norm':
        y=tir_2[i+window_size:i+window_size+1]
    else:
        y=raw_tir_2[i+window_size:i+window_size+1]
    tir_x.append(x)
    tir_y.append(y)

tir_x=np.array(tir_x)
tir_y=np.array(tir_y)

tir_x=np.reshape(tir_x, (tir_x.shape[0],tir_x.shape[1],tir_x.shape[2],tir_x.shape[3],1))  
tir_y=np.reshape(tir_y, (tir_y.shape[0],tir_y.shape[2],tir_y.shape[3],1))  


X= tir_x
y=tir_y

print('X.shape and y.shape:',X.shape, y.shape)

X_train=X[:27]
X_val=X[27:]
y_train=y[:27]
y_val=y[27:]


## modeltraining

from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(
                          monitor='loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,restore_best_weights=True)

callbacks = [earlystop]

model.fit(X_train, y_train ,batch_size=1,epochs=150,callbacks=callbacks,validation_data=(X_val, y_val))
