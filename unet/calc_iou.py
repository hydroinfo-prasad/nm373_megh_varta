# https://www.jeremyjordan.me/evaluating-image-segmentation-models/

import torch
import numpy as np
from PIL import Image
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inp_type', type=str, help='tir/vis/fusion')
parser.add_argument('--cf',type = str, help='1/13')
parser.add_argument('--model_path',type=str,help='path to model')
parser.add_argument('--device',type=str,default='cuda',help='path to model')
parser.add_argument('--save_result',type=int,default='0',help='1/0')



args = parser.parse_args()

# PyTroch version

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded 


def iou_npy(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score

def precision( target, prediction ):
    intersection = np.logical_and( target,prediction )
    den = prediction
    prec = np.sum( intersection ) / np.sum( den )
    return prec
    
    
def recall( target, prediction ):
    intersection = np.logical_and( target, prediction )
    den = target
    recall = np.sum( intersection ) / np.sum( den )
    return recall



if __name__ == '__main__':
    arg = args.inp_type
    cloud_flag = args.cf
    device = args.device
    
    test_index = [0,4,8,12,16]
    iou, prec, rec = [], [], []
    man_iou,man_p,man_r = [],[],[]
    if arg == 'vis':    
        files = os.listdir("./INSAT3D_VIS_India")
        files.sort()
        files = files[8:25]
        test_list = [ files[i] for i in test_index ] 
    elif arg == 'tir':
        files = os.listdir("./INSAT3D_TIR1_India")
        files.sort()
        files = files[8:25]
        test_list = [ files[i] for i in test_index ]
    elif arg == 'fusion':
        files1 = os.listdir("./INSAT3D_VIS_India")
        files1.sort()
        files1 = files1[8:25]
        test_list1 = [ files1[i] for i in test_index ]

        files2 = os.listdir("./INSAT3D_TIR1_India")
        files2.sort()
        files2 = files2[8:25]
        test_list2 = [ files2[i] for i in test_index ]

        test_list = [ (test_list1[i], test_list2[i]) for i in range(5) ]


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torch.load(args.model_path,map_location=device)
    model = model.eval()
    label_list = [f'./../SIH/INSAT_Cloud_Labels/CROPPED_TIFF/CMK_Cropped_{i}.tif' for i in [9,13,17,21,25]]
    
    for ind,img_name in enumerate(test_list):
        if arg != 'fusion':
            if arg == 'vis':
                image = np.array(Image.open(f'./INSAT3D_VIS_India/{img_name}'))[:,:1072]            
            elif arg == 'tir':
                image = np.array(Image.open(f'./INSAT3D_TIR1_India/{img_name}'))[:,:1072]            
        elif arg == 'fusion':

            image1 = np.array(Image.open(f'./INSAT3D_VIS_India/{img_name[0]}'))[:,:1072]
            image1 = image1.astype(np.float)

            image2 = np.array(Image.open(f'./INSAT3D_TIR1_India/{img_name[1]}'))[:,:1072]
            image2 = image2.astype(np.float)

            image = np.array([image1,image2])
            image = image.astype(np.float)
            #image = image.transpose( (1,2,0))

        img_tensor = torch.from_numpy(image)
        #print(img_tensor.shape)

        if arg == 'fusion':
            img_tensor = img_tensor.view((1,2,984,1072))
        else:
            img_tensor = img_tensor.view((1,1,984,1072))


        img_tensor = img_tensor.to(device,dtype=torch.float)
        
        with torch.no_grad():
            out = model(img_tensor)
        output = out.cpu().detach().numpy()
        output = np.squeeze(output)
        output = output > 0
        output = output.astype(float)
        
        
        
        label = np.array(Image.open(label_list[ind]))[:,:1072]
        if cloud_flag == '13':
            label = np.logical_or( label ==1, label ==3 )
        elif cloud_flag == '1' :
            label = label == 1
            
        target = label.astype(np.float)
        
        if args.inp_type == 'vis':
            man_lst,man_prec, man_rec = [], [], []
            for thresh in [0,50,100,200]:
                man_op = image > thresh
                man_op = man_op.astype(np.float)
                man_lst.append(iou_npy(target,man_op))
                man_prec.append( precision(target,man_op) )
                man_rec.append( recall(target,man_op) )
            man_iou.append(man_lst)
            man_p.append(man_prec)
            man_r.append(man_rec)    
            
        if args.inp_type == 'tir':            
            man_lst,man_prec, man_rec = [], [], []
            for thresh in [600,650,700,750,800,850,900]:
                man_op = image < thresh
                man_op = man_op.astype(np.float)
                man_lst.append(iou_npy(target,man_op))
                man_prec.append( precision(target,man_op) )
                man_rec.append( recall(target,man_op) )
            man_iou.append(man_lst)
            man_p.append(man_prec)
            man_r.append(man_rec)    
            
            
        if args.save_result:
            save_dir = f'results_{args.model_path[2:]}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_img = Image.fromarray(output*255)
            save_name = f'{save_dir}/{ind}.tif'
            print(save_name)
            save_img.save(save_name)
        iou.append(iou_npy(target,output))
        prec.append( precision(target,output) )
        rec.append( recall( target,output ) )

        
    iou = np.array(iou)
    prec = np.array(prec)
    rec = np.array( rec )
    man_iou =np.array(man_iou)
    man_p = np.array(man_p)
    man_r = np.array(man_r)
    print(np.mean(man_iou,axis=0) )
    print("precision ", np.mean(man_p,axis=0))
    print( " recall ", np.mean(man_r,axis=0) )
    print(iou)
    print(np.mean(iou), np.mean(prec), np.mean(rec))

            


    


'''
label = np.array(Image.open(f'./../SIH/INSAT_Cloud_Labels/CROPPED_TIFF/CMK_Cropped_13.tif'))[:,:1072]
label = np.logical_or( label==1, label==3 )
target = label.astype(np.float)
label = label.astype(np.bool)
label = np.reshape(label, (1,label.shape[0],label.shape[1]))
print(target)

img = np.array(Image.open(f'./INSAT3D_VIS_India/3DIMG_07NOV2019_0600_L1C_SGP_vis.tif'))[:,:1072]     
                            
    
pred = np.array(Image.open('./out_vis_6000.tif'),dtype=np.bool)
pred = np.reshape(pred,(1,pred.shape[0],pred.shape[1]))                            
prediction = np.array(Image.open('./out_vis_6000.tif'),dtype=np.float)
prediction = prediction > 0
prediction = prediction.astype(float)
print(prediction)

print(img)
#print(prediction)
#prediction = np.ones((label.shape[1],label.shape[2]))
#print(np.sum(prediction-pred))

print(label.shape,pred.shape)
print( iou_npy( target, prediction ) )

for i in range(0,250,10):
    manual_prediction = img >i
    manual_prediction = manual_prediction.astype(np.float)
    print(i, iou_npy(target,manual_prediction))
manual_prediction = img>85
print("85", iou_npy(target,manual_prediction))



#iou, thresholded = iou_numpy(prediction.astype(bool),label)


print(f' iou_score - {iou_score}')
#print(iou, thresholded)
'''
    