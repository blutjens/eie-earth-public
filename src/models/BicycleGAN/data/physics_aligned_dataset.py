import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch as th

class PhysicsAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset that furthermore ingests segmented data from physics models.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of different folders train_AB, test_AB and val_AB.
    During test time, you need to prepare a directory '/path/to/data/test'.
    Example of use:
    python ./BicycleGAN/train.py --model physics_informed_bicycle_gan --name xBD_256_physics_informed_v01 --gpu_ids 2,3 --direction APtoBP --dataset_mode physics_aligned --dataroot ./xBD/smalltiles/train_AB/ --niter 1 --niter_decay 0 --save_epoch_freq 1 --display_id -1 --load_size 1024 --crop_size 256 --batch_size 2
  
    Example of run (working 24jul en *_tesla VM).
    python ./BicycleGAN/train.py --model physics_informed_bicycle_gan --name xBD_BicycleGAN_256_physics_informed_v02_24jul --gpu_ids 0,1,2,3 --direction APtoBP --dataset_mode physics_aligned --dataroot ./xBD/bigtiles/ --niter 1 --niter_decay 0 --save_epoch_freq 1 --display_id -1 --load_size 1024 --crop_size 256 --batch_size 2  --conditional_D --dataroot_A_masks ./xBD/bigtiles/masks_v1  --dataroot_B_masks ./xBD/bigtiles/masks_v1
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase,'_AB/').replace('/_', '_')  # get the image directory
        print('=====',opt.phase, self.dir_AB, opt.dataroot_A_masks)
        # get the image physics super directory (train_mask, val_mask and test_mask folders must live here)
        # masks_folder_data_A = glob.glob(self.dir_A+"\\*mask.jpg") # mask
        if opt.phase == 'train':
            self.dir_AP = os.path.join(opt.dataroot_A_masks, 'train_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, 'train_mask')
        elif opt.phase == 'test':
            self.dir_AP = os.path.join(opt.dataroot_A_masks, 'test_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, 'test_mask')
        elif opt.phase == 'val':
            self.dir_AP = os.path.join(opt.dataroot_A_masks, 'val_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, 'val_mask')
        else:
            print(" WRONG opt.phase option: ", opt.phase)
        # self.dir_AP = os.path.join(opt.dataroot_masksA, 'SegmMasks')  
        # self.dir_BP = os.path.join(opt.dataroot_masksB, 'SegmMasks')  # get the image physics super directory (train/, val/, test/)
        # #assert 'mask' in self.dir_AP and 'mask' in self.dir_BP, 'Make sure you have placed your segmentation mask images in  \
        #    a folder containing \'mask\' on it and pass these as parameters'
        print('dir_AB: {}. dir_AP could be the same as dir_BP if no pre and post masks available: {}\n{}'.format(self.dir_AB, self.dir_AP, self.dir_BP))
        try:
            if not os.path.exists(self.dir_AP):
                raise Exception("Directory not accessible: {}. Train_mask, val_mask and test_mask folders must live here. \
                Please make sure you have downloaded both data and its segmentation masks. \n Example \
                Here xBD data and masks are in /home/jupyter/xBD.\n Download the data: \n \
                gsutil -m cp -r gs://eie_floods/masks_v1 /home/jupyter/xBD/bigtiles".format(self.dir_AP))
            if not os.path.exists(self.dir_BP):
                raise Exception("Directory not accessible: {}. Train_mask, val_mask and test_mask folders must live here. \
                Please make sure you have downloaded both data and its segmentation masks. \n Example \
                Here xBD data and masks are in /home/jupyter/xBD.\n Download the data: \n \
                gsutil -m cp -r gs://eie_floods/masks_v1 /home/jupyter/xBD/bigtiles".format(self.dir_BP))
        finally:
            print('Successfully read images and physics related segmentation masks ', self.dir_AB, self.dir_AP,self.dir_BP)

        # toDo meantime: (right now the segm. flood masks are the same for input and output images (A and B) and they correspond always to post flood images)
        # the 4th channel is the input condition to the conditional discriminator (default bicyclegan isnt conditional).
        # self.dir_AP = self.dir_AB # get the image physics super directory (train/, val/, test/)
        # self.dir_BP = self.dir_AB  # get the image physics super directory (train/, val/, test/)
        print('but while data is ready, it is for now: ', self.dir_AP, opt.max_dataset_size)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size)) # get image paths
        self.AP_paths = sorted(make_dataset(self.dir_AP, opt.max_dataset_size))  # get image mask paths
        self.BP_paths = sorted(make_dataset(self.dir_BP, opt.max_dataset_size))  # get image mask paths

        assert self.opt.load_size >= self.opt.crop_size, "crop_size should be smaller than the size of loaded image"
        # Extend to handle also Physics channel in the target image
        assert self.opt.direction == 'APtoBP' or self.opt.direction == 'BPtoAP', 'Valid options for this model are APtoBP or BPtoAP'
        assert len(self.AB_paths) == len(self.AP_paths), 'Number of images in training data --dataroot must be equal to the number of images \
            in the physics masks (--dataroot_A_masks and also in --dataroot_B_masks): AB: {}, AP: {}, BP: {}'.format(len(self.AB_paths), len(self.AP_paths), len(self.BP_paths))
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BPtoAP' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BPtoAP' else self.opt.output_nc
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, P, A_paths and B_paths and P_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # We assume the masks are placed in different but same level as train_A folder 
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        #if self.opt.direction == 'APtoBP':
        AP_path = self.AP_paths[index]
        P_pre = Image.open(AP_path).convert('LA') #LA mode has luminosity (brightness) and alpha 
        #else: #ToDo, we can avoid creating either of them in each run as it is not running AtoB and BtoA always (but BzB and zBz).
        BP_path = self.BP_paths[index]
        BP = Image.open(BP_path).convert('LA')
        P_post = Image.open(BP_path).convert('LA') 
         
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        p_w, p_h = P_pre.size
        assert w2==p_w and h==p_h, 'Dimensions of masks do not match dimensions of input images, i.e. half of AB size: w: {}, h: {}. Physics masks P_pre: p_w: {}, p_h: {}.'.format(w2, h, p_w, p_h)
        #fpr fast testing
        # P_pre = P_pre.crop((0, 0, w2, h)) 
        # P_post = P_post.crop((w2, 0, w, h))
         
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=self.input_nc == 1)
        B_transform = get_transform(self.opt, transform_params, grayscale=self.output_nc == 1)
        P_transform = get_transform(self.opt, transform_params, grayscale=True)
        
        A = A_transform(A)
        B = B_transform(B)
        # physics informing channel image with segmentation masks pre and post (flood) event 
        P_pre = P_transform(P_pre)
        P_post = P_transform(P_post)

        # merge A with the physical mask of A (AP)    #print('AP before: ', P_pre.size(), P_post.size()) # (1,256,256)
        AP = th.cat((A, P_post), dim=0)
        BP = th.cat((B, P_pre), dim=0)
        
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'AP': AP, 'AP_paths': AP_path, 'BP': BP, 'BP_paths': BP_path, 'P_pre': P_pre, 'P_post': P_post}

    def random_split(self):
        # toDo
        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
