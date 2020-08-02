# Code for creating all the TIF images to JPEG of the working directory

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from PIL import Image

def convert_jpeg(current_img,name):
    # Data for plotting

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(current_img,cmap='gray')

    ax.set_title(' Date {} Time'.format(name))
    ax.axis('off')

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #print(image.shape)
    plt.imsave( name+'.jpeg',image,format='jpeg',cmap='gray',dpi=300)
#    return image

# path = "INSAT3D_TIR1_India\INSAT3D_TIR1_India\*.tif"
path = '*.tif'
name_list = glob.glob(path)
name_list.sort()

input_imgs = glob.glob(path)
imgs = [Image.open(img) for img in sorted(input_imgs)]
no_images = len(input_imgs)
h,w = imgs[0].size
input_array = np.zeros((w,h ,no_images))

for i in range(no_images):
    print(name_list[i][7:20])
    convert_jpeg(imgs[i],name_list[i][7:20]) #  Here the indices 7 to 20 change according to naming style
