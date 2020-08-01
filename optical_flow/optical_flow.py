import os
import random
import numpy as np
from PIL import Image
from rainymotion.models import *
import time

import sys
sys.path.append("/home/prasad/")
from utils.pre_processing import get_seq_data


start = time.time()

dir_path = "./../pytorch-unet/INSAT3D_VIS_India"

inp_seq = 3
pred_frame = 1
inp,target = get_seq_data(dir_path,inp_seq,pred_frame)
inp = inp[22:]
target = target[22:]
#for elems in zip(inp,target):
#    print(elems[0],elems[1])
# sample code 
img_list = []


    
window = inp_seq
pred_len = pred_frame
model = Dense()

mean1, mean2 = [],[]
for i in range( len(target) ):
    if( i%5==0 ):
        print(i)
    img = []
    for j in range(window):
        img.append( np.array(Image.open( inp[i][j] ))[:,:1072] )
    
    img = np.array(img)
    
    model.input_data = img
    model.lead_steps = 1;
    nowcast = model.run()
    gt = []
    for j in range(pred_len):
        gt.append( np.array( Image.open( target[i][j] ) )[:,:1072] )
    gt = np.array(gt)
    latest = np.array( img[2] )
    
    mean1.append( ((gt - nowcast)**2).mean() )
    mean2.append( ((latest-gt)**2).mean() )

mean1 = np.array(mean1)
mean2 = np.array(mean2)
print( mean1 )
print( mean2 )
print( mean1.mean(), mean2.mean() )
print(mean1.shape,mean2.shape)
print( "program took - ", time.time()-start, " s." )
