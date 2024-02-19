#Two modes, xBD and xBD_full (only floods vs all)
#create xBD_full_augmented with n=4
#create xBD_flood_augmented with n=10

from PIL import Image
import os
import glob
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import colorsys
import random
import numpy as np
from time import perf_counter

def images_paths(path):
    '''
    Returns paths for triplets of path/train_A, path/train_B, and path/train_mask
    '''
    train_A = glob.glob(path +'/train_A/*.png')
    train_A.sort()
    train_B = glob.glob(path +'/train_B/*.png')
    train_B.sort()
    train_mask = glob.glob(path +'/train_mask/*.png')
    train_mask.sort()
    im_paths = [train_A, train_B, train_mask]
    
    #Transpose the list of lists
    im_paths = list(map(list, zip(*im_paths)))
    
    return im_paths


def load_triplet(triplet_path):
    
    pre = np.array(Image.open(triplet_path[0]))
    post = np.array(Image.open(triplet_path[1]))
    mask = np.array(Image.open(triplet_path[2]))
    
    triplet = {'pre':pre,'post':post,'mask':mask}
    return triplet

    
def save_triplet(triplet, save_path, im_id):
    pre = Image.fromarray(triplet['pre'])
    post = Image.fromarray(triplet['post'])
    mask = Image.fromarray(triplet['mask'])
    
    pre.save(save_path+'/train_A/'+im_id[0]+'.png')
    post.save(save_path+'/train_B/'+im_id[1]+'.png')
    mask.save(save_path+'/train_mask/'+im_id[2]+'.png')
    
    
def random_rotate(seed, triplet, rotate_angle=180):
    random.seed(seed)
    angle = random.uniform(-rotate_angle, rotate_angle)
    
    
    triplet['pre'] = rotate(triplet['pre'], angle, reshape=False, cval=0)
    triplet['post'] = rotate(triplet['post'], angle, reshape=False, cval=0)
    triplet['mask'] = rotate(triplet['mask'], angle, reshape=False, cval=255)
    
    return triplet

def elastic_transform(image, alpha, sigma, seed, cval):
    """
    Adapted for RGB images from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    channels = [image[:,:,0],image[:,:,1],image[:,:,2]] #
    channels = np.split(image, 3, axis=2)
    for ch_num, channel in enumerate(channels):
        channel = np.squeeze(channel)
        assert len(channel.shape)==2

        random_state = np.random.RandomState(seed)

        shape = channel.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=cval) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=cval) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        image[:,:,ch_num] = map_coordinates(channel, indices, order=1, cval=cval).reshape(shape)

    return image

def random_elastic_transform(seed, triplet):
    random.seed(seed)
    alpha = random.uniform(5, 20)
    sigma = random.uniform(1, 4)
    
    triplet['pre'] = elastic_transform(triplet['pre'], alpha, sigma, seed=seed, cval=0)
    triplet['post'] = elastic_transform(triplet['post'], alpha, sigma, seed=seed, cval=0)
    triplet['mask'] = elastic_transform(triplet['mask'], alpha, sigma, seed=seed, cval=255)
    
    return triplet

def random_flip(seed,triplet):
    
    def flip(im, v, h):
        if v:
            im = np.flip(im, axis=0)
        if h:
            im = np.flip(im, axis=1)
        return im
    
    random.seed(seed)
    vertical = random.getrandbits(1)
    horizontal  = random.getrandbits(1)
    
    triplet['pre'] = flip(triplet['pre'], vertical, horizontal)
    triplet['post'] = flip(triplet['post'], vertical, horizontal)
    triplet['mask'] = flip(triplet['mask'], vertical, horizontal)
    
    return triplet

def random_color_jitter(seed, triplet):
    
    def color_jitter(im, hue, sat, val):
        rgb_to_hsv=np.vectorize(colorsys.rgb_to_hsv)
        hsv_to_rgb=np.vectorize(colorsys.hsv_to_rgb)
        
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        h, s, v=rgb_to_hsv(r, g, b)
        
        h=h + hue
        s=s + sat
        v=v + val
        
        r, g, b=hsv_to_rgb(h, s, v)
        im_out = np.stack([r,g,b],axis=2).astype("uint8")
        im_out = np.where(im==0,0,im_out)
        return im_out
    
    random.seed(seed)
    hue = random.uniform(-0.05, 0.05)
    saturation = random.uniform(-0.05, 0.05)
    value = random.uniform(-0.05, 0.05)
    
    triplet['pre'] = color_jitter(triplet['pre'], hue, saturation, value)
    triplet['post'] = color_jitter(triplet['post'], hue, saturation, value)
    
    #Color jitter does not apply to mask
    #triplet['mask'] = triplet['mask'] 
        
    return triplet

def augment_dataset(input_dir, output_dir, n, start, end, elastic=True, color_jitter=True, rotate_angle=180):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir+'/train_A'):
        os.mkdir(output_dir+'/train_A')
    if not os.path.exists(output_dir+'/train_B'):
        os.mkdir(output_dir+'/train_B')
    if not os.path.exists(output_dir+'/train_mask'):
        os.mkdir(output_dir+'/train_mask')
    
    triplet_paths = images_paths(input_dir)[start:end]
    # Start the stopwatch / counter 
    t1 = perf_counter()  

    for sample, triplet_path in enumerate(triplet_paths):
        t2 = perf_counter()  
        triplet = load_triplet(triplet_path)
        im_ids=[]
        for im_path in triplet_path:
            im_ids.append(im_path.split('/')[-1][:-4])
        
        for i in range(n):
            seed = sample*1000+i
            num = str(i).zfill(2)
            im_ids_num = [s + '_' + num for s in im_ids]
            

            augmented_triplet = random_flip(seed, triplet)
            if elastic:
                augmented_triplet = random_elastic_transform(seed, augmented_triplet)
            if color_jitter:
                augmented_triplet = random_color_jitter(seed, augmented_triplet)
            augmented_triplet = random_rotate(seed, augmented_triplet, rotate_angle)
            
            save_triplet(augmented_triplet, output_dir, im_ids_num)
        
        print("Sample {} augmented in {} seconds".format(sample, perf_counter()-t2))
        if sample % 20 == 0 and sample is not 0:
            print("Average time per sample is {} seconds".format((perf_counter()-t1)/sample))
    return start

