import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
#%matplotlib inline

import sys
from pre_processing import get_seq_data


dir_paths = ["./INSAT3D_TIR1_India", "./INSAT3D_VIS_India" ]
print_format = ['TIR','VIS']

inp_seq = 3
pred_frame = 1

# for final output
last_sequence = [ ['3DIMG_07NOV2019_2230_L1C_SGP.tif','3DIMG_07NOV2019_2300_L1C_SGP.tif','3DIMG_07NOV2019_2330_L1C_SGP.tif'],
                    ['3DIMG_07NOV2019_2230_L1C_SGP_vis.tif','3DIMG_07NOV2019_2300_L1C_SGP_vis.tif','3DIMG_07NOV2019_2330_L1C_SGP_vis.tif'] ]



# values calculated on training set
mn,mx = 387.0, 948.0
vmean, vstd = 68.5147950474417, 71.67788656008055

model_paths = ["./tir_inp3_ds4_stand.ckpt", "./vis_inp3_1.ckpt"]
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    



def pre_process(inp,iformat = 'TIR'):
    
    if( iformat=='TIR' ):
        inp = (inp-mn)/(mx-mn)
    else:
        inp = (inp-vmean)/vstd
        
    inp = np.expand_dims(inp,axis=0)
    inp = torch.from_numpy(inp)
    inp = inp.to(device,dtype=torch.float)

    return inp

def post_process(inp,iformat='TIR'):

    if( iformat=='TIR' ):
        inp = inp*(mx-mn) + mn
    else:
        inp = inp*vstd + vmean
    inp = inp.detach().numpy()
    inp = np.squeeze(inp)[:984,:1074]
    return inp

tir_mse = []
for i,dir_path in enumerate(dir_paths):
    inp,target = get_seq_data( dir_path )
    
    '''
    if print_format[i] == 'TIR':
        inp,target = inp[-5:], target[-5:]
    else:
        inp, target = inp[10:15], target[10:15]
    '''
    print(print_format[i])
    model = torch.load(model_paths[i],map_location=device)
    model = model.to(device)
    model.eval() 

    for k in range( len(target) ):
        # get images in numpy array
        img,gt = [],[]
        for j in range(inp_seq):
            img.append( np.array(Image.open(  inp[k][j] ) ) ) 
            #print(inp[k][j])

        gt = np.array( Image.open(target[k][0]) ) 
        latest = np.array( Image.open(   inp[k][-1] ) )    # most recent image for prediction at 2330              
        #print(target[k][0])

        img, gt = np.array(img), np.array(gt)
        img = pre_process(img,print_format[i])

        output = model(img).cpu()    
        output = post_process(output,iformat=print_format[i])

        unet_mse = ((output-gt)**2).mean()
        unet_mae =  np.abs( output-gt).mean() 

        bmse = ((latest-gt)**2).mean()
        bmae = np.abs(latest-gt).mean()
        
        if print_format[i] == 'TIR':
            tir_mse.append(unet_mse)
        
        print( "Baseline MSE - ", bmse, " Unet MSE - ", unet_mse, "Improvement - ", (bmse-unet_mse)*100/bmse, "%" )
        print( "Baseline MAE - ", bmae, " UNet MAE - ", unet_mae , "Improvement - ", (bmae-unet_mae)*100/bmae, "%")
        '''   
        Image.fromarray( np.squeeze(output) ).save( f"{print_format[i]}_results/unet_pred_{target[k][0][target[k][0].rfind('/')+1:]}_{print_format[i]}.tif" )
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(gt,cmap='Greys')
        tind = -16 if print_format[i]=='TIR' else -20                                           
        ax.set_title(f"GT {print_format[i]} at {target[k][0][tind:tind+4]}")
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(np.squeeze(output),cmap='Greys')
        ax.set_title(f"Predicted {print_format[i]} at {target[k][0][tind:tind+4]}" )
    
        plt.show()
        '''   
print( tir_mse )        
print(np.array(tir_mse).mean())