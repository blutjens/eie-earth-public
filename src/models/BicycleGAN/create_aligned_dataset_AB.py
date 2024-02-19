import numpy as np
import sys
import os
import numpy as np
from PIL import Image
from os.path import isfile, join
import csv


def bigtiles_to_smalltiles(bigtiles="./xBD/bigtiles/", small_path="./xBD/smalltiles/",output_size=256):
    folders = ['train_A', 'train_B', 'train_AB', 'test_A', 'test_B', 'test_AB', 'val_A', 'val_B', 'val_AB']
    for folder in folders:
        if not os.path.exists(small_path+folder):
            os.makedirs(small_path+folder)
        
    def crop_tiles(image):
        image = np.array(image)
        width = image.shape[0]
        N = width/output_size
        h_tiles = np.split(image, N)
        
        images = []
        for tile in h_tiles:
            images.append(np.split(tile, N, axis=1))
        images = [item for sublist in images for item in sublist] #flatten nested list
        return images
    
    def load_crop_save(list_files, path):
        for i in range(len(list_files)):
            image = Image.open("./xBD/bigtiles/"+path+list_files[i])
            images = crop_tiles(image)
            for num , im in enumerate(images):
                name = list_files[i][:-4]+str(num).zfill(2)+'.png'
                new_im = Image.fromarray(im)
                new_im.save('./xBD/smalltiles/'+path+name)
    
    for dataset in ['train','test','val']:
        A_dir = bigtiles + dataset + '_A/'
        B_dir = bigtiles + dataset + '_B/'

        train_A_list = [f for f in os.listdir(A_dir) if isfile(join(A_dir, f))]
        train_A_list.sort()
        print("Cropping "+ dataset+ " A")
        load_crop_save(train_A_list, dataset+'_A/')

        train_B_list = [f for f in os.listdir(B_dir) if isfile(join(B_dir, f))]
        train_B_list.sort()
        print("Cropping "+dataset+" B")
        load_crop_save(train_B_list, dataset+'_B/')
    print('Done with '+dataset)


def extension(filename):
    return filename.split('.')[-1]

def valid_extension(filename):
    return extension(filename) in set(['jpg','png','jpeg'])

def A_and_B_to_AB(path="./xBD/bigtiles/"):
    print('Make sure files are named in a way that sorting them alphabetically \
        both in *_A and *_B folders produces the same order (i.e., ideally, \
        files are named equally but placed in different *_A and *_B folders')
        
    for folder in ['train','test','val']:
        A_dir = path + folder + '_A/'
        B_dir = path + folder + '_B/'
        AB_dir = path + folder + '_AB/'
        print('Generating ',AB_dir)
        try:
            os.mkdir(AB_dir)
        except FileExistsError:
            pass
        train_A_list = [f.strip() for f in os.listdir(A_dir) if isfile(join(A_dir, f)) and valid_extension(join(A_dir, f))]
        train_A_list.sort()

        train_B_list = [f.strip() for f in os.listdir(B_dir) if isfile(join(B_dir, f)) and valid_extension(join(B_dir, f))]
        train_B_list.sort()

        for img_path_A, img_path_B in zip(train_A_list, train_B_list):
            # assert img_path_A == img_path_B, 'Make sure files are named in a \
            # way that sorting them alphabetically both in *_A and *_B folders \
            # produces the same order (i.e., ideally, files are named equally but \
            # placed in different *_A and *_B folders: {}\n{}'.format(img_path_A, img_path_B)
            print('Generating {} image AB for {}'.format(folder,img_path_A))
            images = [Image.open(x) for x in [A_dir + img_path_A, B_dir+img_path_B]]
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]

            new_im.save(path+folder+'_AB/'+img_path_A)
        print('Done with ' +folder)
    
  

if __name__ == '__main__':    
    bigtiles_to_smalltiles()
    # Warning, if this error is gotten, for safety, remove it, some are corrupted: \
    # PIL.UnidentifiedImageError: cannot identify image file. \
    # e.g.: training-set_train_images_hurricane-florence_00000401_post_disaster_fake_B.png')
    A_and_B_to_AB("./xBD/bigtiles/")