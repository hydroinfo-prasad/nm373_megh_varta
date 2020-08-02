import sys
sys.path.append("./../utils/")
from pre_processing import get_seq_data

import numpy as np
import torch
from PIL import Image

path = "./INSAT3D_TIR1_India/"    
inp, target = get_seq_data( path, inp_seq=3,pred_frame=1 )
num_inp = len(target)
ind = int(num_inp*0.7)

fmstd = []

for image in inp[0]:
    fmstd.append( np.array( Image.open(image), dtype= np.double ) )
for image in target[:ind]:
    fmstd.append( np.array(Image.open( image[0] ),dtype = np.double ) )

    
fmstd = np.array(fmstd,dtype=np.double)                 
mean_value = np.mean(fmstd)
std_value =  np.std(fmstd)
fmstd = []                 

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = torch.load('check.ckpt')
model = model.to(device)
model.eval() 
seq_frame = 3
pred_frame = 1

mse = []
    
for elems in zip( inp[ind:], target[ind:] ):
    seq, targ = [],[]
    for i in range(seq_frame):
        image = np.array(Image.open(elems[0][i]))[:,:1072]
        image = image.astype(np.double)
        seq.append(image)
    seq = np.array(seq,dtype=np.double)
    seq = (seq - mean_value)/std_value
    seq = np.expand_dims(seq,axis=0)
    seq = torch.from_numpy(seq)
    for i in range(pred_frame):
        image = np.array(Image.open(elems[1][i]))[:,:1072]
        image = image.astype(np.double)
        targ.append(image)
    targ = np.array(targ)
    #target = torch.from_numpy(target)
    #target = (target - self.mean)/self.std
    seq = seq.to(device, dtype=torch.float)
    output = model(seq).detach().cpu()
    output = output*std_value + mean_value
    output = np.squeeze(output.numpy())
    mse.append( ((output-targ)**2).mean() )
                    
            
mse = np.array(mse)
print(mse)
print(mse.mean())


