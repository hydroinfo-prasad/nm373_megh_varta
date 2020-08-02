import numpy as np
from PIL import Image
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
#sys.path.append("./../pytorch-unet/")
#sys.path.append("./../utils/")
#from pre_processing import get_seq_data


def convert_jpeg(current_img,name,tm):
    # Data for plotting

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(current_img,cmap='gray')

    ax.set_title(f' {tm} '.format(name))
    ax.axis('off')

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(image.shape)
    plt.imsave( name+'.jpeg',image,format='jpeg',cmap='gray',dpi=300)
#    return image


tm = sys.argv[1]
dir_path = "./INSAT3D_TIR1_India/"
data = os.listdir(dir_path)
data.sort()
data = [elem for elem in data if int(elem[18:20]) % 30 == 0]
ind = 0
print("time",tm)
for i,fl in enumerate(data):
	if int( fl[16:20]) == int(tm):
		ind = i
		break
print(ind)
inp_files = [[ "./INSAT3D_TIR1_India/"+data[ind-2], "./INSAT3D_TIR1_India/"+data[ind-1], "./INSAT3D_TIR1_India/"+data[ind]   ]]
target_files = [["./INSAT3D_TIR1_India/"+data[ind+1]]]


#inp_files, target_files = get_seq_data(dir_path,inp_seq=3,pred_frame=1)
#inp_files = inp_files[-3:]
#target_files = target_files[-3:]
print(len(target_files))
'''
for elems in zip(inp_files,target_files):
    for item in elems[0]:
        print(item)
    
    print(elems[1])
    print()
'''
baseline = []
unet = []
# values calculated on training set
mean, std = 652.7621795090157, 110.78180624410955
dmean, dstd =  647.3035566765255, 107.2158170217211
nmean,nstd =  655.0820899218569, 111.28136750965064
vmean,vstd =  68.5147950474417, 71.67788656008055
mn,mx = 387.0, 948.0

#model_path = "./../pytorch-unet/check.ckpt"
model_path = "./tir_inp3_ds4_stand.ckpt"


device = torch.device('cpu')#'cuda:2' if torch.cuda.is_available() else 'cpu')
#print(device)
model = torch.load(model_path,map_location=device)
model = model.to(device)
model.eval() 


for i in range( len(target_files) ):
    latest = np.array( Image.open( inp_files[i][-1] ) )
    ground_truth = np.array( Image.open(  target_files[i][0]  ) )
    bmse = ((latest-ground_truth)**2).mean()
    baseline.append(bmse)
    
    inp = []
    for nind,files in enumerate(inp_files[i]):
        print(files)
        img = Image.open( files  )
        convert_jpeg(np.array(img),str(nind+1),files[-16:-12])
        inp.append(  np.array( img ) )
        
    inp = np.array(inp)
    inp = (inp-mn)/(mx-mn)
    #inp = (inp-vmean)/vstd
    print(target_files)
    inp = np.expand_dims(inp,axis=0)
    inp = torch.from_numpy(inp)
    inp = inp.to(device,dtype=torch.float)
    
    output = model(inp).cpu()
    output = output*(mx-mn) + mn
    #output = output*vstd+vmean
    output = np.squeeze(output.detach().numpy())
    output = output[:984,:1074]
    convert_jpeg(output,"4","prediction")
    unet_mse = ((output-ground_truth)**2).mean()
    unet.append(unet_mse)

    
baseline = np.array(baseline)
bmse = baseline.mean()
print(baseline)
print( " Baseline MSE -  ", bmse )

unet = np.array(unet)
print(unet)
unet_mse = unet.mean()
print( "UNet MSE - ", unet_mse )
Image.fromarray(output).save("unet_pred.tiff")


    