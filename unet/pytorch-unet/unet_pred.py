import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torch
import torch.nn as nn
import pytorch_unet
#import unet_ds
from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

import torch.nn as nn
import sys
print(sys.path)
sys.path.append("./../utils/")
from pre_processing import get_seq_data

from torch.utils.tensorboard import SummaryWriter
import torchvision


preprocess = 'stand'


class SimDataset(Dataset):
    def __init__(self,inp_list, target_list, mean_value, std_value,mn,mx, transform=None, seq_frame=3, pred_frame=1):
        self.inp_list = inp_list
        self.target_list = target_list
        self.transform = transform
        self.seq_frame = seq_frame
        self.pred_frame = pred_frame
        self.mean = mean_value
        self.std = std_value
        self.min_value = mn
        self.max_value = mx
        
    def __len__(self):
        return  len(self.target_list)
    
    def __getitem__(self, idx):
        inp, target = [],[]
        for i in range(self.seq_frame):
            #print(self.inp_list[idx][i])
            image = np.array(Image.open(self.inp_list[idx][i]))#[:,:1072]#[480:1464,1200:2280]#
            image = image.astype(np.double)
            inp.append(image)
        inp = np.array(inp,dtype=np.double)
        
        if preprocess == 'norm':
            inp = (inp-self.mean)/self.std
        elif preprocess == 'stand':
            inp = (inp - self.min_value)/(self.max_value-self.min_value) 
            
        for i in range(pred_frame):
            image = np.array(Image.open(self.target_list[idx][i]))#[:,:1072]#[480:1464,1200:2280]#
            image = image.astype(np.double)
            target.append(image)
        target = np.array(target)
        if preprocess == 'norm':
            target = (target - self.mean)/self.std
        elif preprocess == 'stand':
            target = (target - self.min_value)/( self.max_value - self.min_value )
        #if self.transform:
            #print(inp.shape)
            #inp = self.transform(inp)
            #target = self.transform(target)
        
        #print(inp,target)
        return [inp,target]

'''

path = "/home/prasad/SIH/oct/tir/"    
inp, target = get_seq_data( path, inp_seq=3,pred_frame=1 )

path2 = "/home/prasad/SIH/nov_2019_data/tir/"
inp2, target2 = get_seq_data( path2,inp_seq=3,pred_frame=1 )


cut_ind = 170
print( inp2[cut_ind] )
inp.extend( inp2[:cut_ind] )
target.extend( target2[:cut_ind] )

# to check sequences
file = open("check_seq.txt","w")

for elems in zip( inp,target ):
    for iseq in elems[0]:
        file.write(f"{iseq}\n")
    for oseq in elems[1]:
        file.write(f"{oseq}\n")
    file.write("\n")

file.close()

'''

path = "./INSAT3D_TIR1_India/"
inp_seq=3
pred_frame = 1

inp, target = get_seq_data( path, inp_seq,pred_frame )
print(len(target))

file = open("check_seq.txt","w")

for elems in zip( inp,target ):
    for iseq in elems[0]:
        file.write(f"{iseq}\n")
    for oseq in elems[1]:
        file.write(f"{oseq}\n")
    file.write("\n")

file.close()
    
#  train val split of 80/20
num_inp = len(target)
ind = int(num_inp*0.9)

fmstd = []

pixel_sum, pixel_num,pixel_sum_square = 0,0,0
mn,mx = 100000,0

for image in inp[0]:
    img = np.array( Image.open(image), dtype= np.double )#[480:1464,1200:2284]
    pixel_sum += np.sum(img)
    pixel_sum_square += np.sum( np.square(img))
    pixel_num += img.size
    mn = min( np.min(img),mn )
    mx = max( np.max(img),mx )
    
for image in target[:ind]:
    img = np.array(Image.open( image[0] ),dtype = np.double )#[480:1464,1200:2284]
    pixel_sum += np.sum(img)
    pixel_sum_square += np.sum(np.square(img))
    pixel_num += img.size
    mn = min( np.min(img),mn )
    mx = max( np.max(img), mx )

    
    
mean_value = pixel_sum/pixel_num
std_value =  np.sqrt(pixel_sum_square/pixel_num - np.square(mean_value))
fmstd = []                 

#gmean = mean_value.to(device)
#gstd = std_value.to(device)


#mean_value,std_value = 647.3035566765255, 107.2158170217211
print(ind,num_inp,mean_value,std_value,mn,mx)

# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize( mean = [mean_value],std=[std_value])
])


train_set = SimDataset(inp[:ind], target[:ind], mean_value, std_value,mn,mx, trans,inp_seq,pred_frame)
val_set = SimDataset(inp[ind:], target[ind:], mean_value, std_value,mn,mx,  trans,inp_seq,pred_frame)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 1

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet.UNet(pred_frame,inp_channel=inp_seq)
#model = unet_ds.UNetDS5(pred_frame,inp_channel=inp_seq)

#model = torch.load("mosdac.ckpt")


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    preds = np.squeeze(output.detach().cpu().numpy())
    if preprocess == 'norm':
        preds = preds*std_value + mean_value
    elif preprocess == 'stand':
        preds = preds*(mx-mn) + mn
    return preds

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 20))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        if idx<3:
            img = np.squeeze( images.cpu().numpy() )[idx]
            plt.imshow(img )
            ax.set_title("Input \n ")
        elif idx == 3:
            img = np.squeeze(labels.cpu().numpy())
            plt.imshow( img )
            ax.set_title("GT \n ")     
        else:
            plt.imshow(  preds ) 
            labels = labels.cpu().numpy()
            if preprocess == 'norm':
                labels = labels*std_value + mean_value
            elif preprocess == 'stand':
                labels = labels*(mx-mn) + mn
            mse = ((preds[:984,:1074]-labels)**2).mean()
            print(mse)
            ax.set_title(F"Prediction \n MSE - {mse}")
        
            
        
    return fig


#summary(model, input_size=(3, 256, 280))
# source https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).to(device)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    """
    mse = nn.MSELoss()
    mse_loss = mse(pred[:,:,:984,:1074],target)
    
    perceptual_loss = VGGPerceptualLoss()
    loss = perceptual_loss( pred[:,:,:984,:1074], target ) + 500*mse_loss
    
    metrics['mse'] +=  mse_loss.data.cpu().numpy()* target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples  ))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, writer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                #for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()
                    
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                '''
                if(epoch_samples%300 == 0):
                    print_metrics(metrics,epoch_samples,phase)
                    writer.add_scalar(' training loss ', metrics['loss']/(epoch_samples + 1e-6),epoch * len(dataloaders[phase]) + epoch_samples)
                    writer.add_figure(" prediction vs actual ", plot_classes_preds(model,inputs,labels) )
                    writer.add_scalar( " Learning Rate ", optimizer.param_groups[0]['lr'] )
                '''
                epoch_samples += inputs.size(0)
                
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
            

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            
            if phase == 'train':
                print("logged")
                writer.add_scalars('loss', {'training loss': metrics['loss']/(epoch_samples)   },epoch * len(dataloaders[phase]) + epoch_samples)
                writer.add_scalars('MSE', {'training loss': metrics['mse']/(epoch_samples)   },epoch * len(dataloaders[phase]) + epoch_samples)
                writer.add_figure(f" prediction vs actual ", plot_classes_preds(model,inputs,labels), epoch )
                writer.add_scalar( " Learning Rate ", optimizer.param_groups[0]['lr'], epoch*len(dataloaders[phase]) + epoch_samples)
            if phase == "val":
                writer.add_scalars('loss', {'validation loss': metrics['loss']/(epoch_samples)},epoch * len(dataloaders['train']) + ind)
                writer.add_scalars('MSE', {'validation loss': metrics['mse']/(epoch_samples)},epoch * len(dataloaders['train']) + ind)
                
            

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,optimizer,scheduler



writer = SummaryWriter('/nfs/151/gpu/gunji/SIH/pytorch-unet/logs/tir_inp3_ds4_stand_percep_mse')


num_class = 1

#model = pytorch_unet.UNet(num_class).to(device)
#model = model.to(device)

'''
for param in model.parameters():
    param.requires_grad = False

for param in model.conv_last.parameters():
    param.requires_grad = True
'''
model = model.to(device)


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.75)

model,optimizer_ft, exp_lr_scheduler = train_model(model, optimizer_ft, exp_lr_scheduler, writer,num_epochs=300)

torch.save(model,"tir_inp3_ds4_stand_percep_mse.ckpt")
writer.close()
