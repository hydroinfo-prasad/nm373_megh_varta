import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from PIL import Image
from pre_processing import get_seq_data

def animate_for_gif(current_img,name):
    # Data for plotting

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(current_img,cmap='gray')

    ax.set_title(f" {name[:-5]} {name[-4:]}")
    ax.axis('off')

    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #print(image.shape)

    return image

path = './TIR_results/*.tif'
#path = "./INSAT3D_TIR1_India"
#_,name_list = get_seq_data(path)
#name_list = [elems[0] for elems in name_list]
name_list = glob.glob(path)
name_list.sort()
print(len(name_list))
input_imgs = glob.glob(path)
imgs = [Image.open(img) for img in name_list]
#no_images = len(name_list)
no_images = len(input_imgs)
h,w = imgs[0].size
input_array = np.zeros((w,h ,no_images))
for i in range(no_images):
#    input_array[:,:,i] = np.concatenate(np.array(imgs[i]), axis=2)
#    print(i)
    input_array[:,:,i] = (np.array(imgs[i]))
    
data = np.copy(input_array)
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
print(name_list[0][-34:-20])
imageio.mimsave('pred_tir_ff.gif', [animate_for_gif(data[:,:,i],name_list[i][-34:-20]) for i in range(no_images)], fps=1)
print('Done')