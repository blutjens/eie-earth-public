import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch as th

class PhysicsAlignedBinDataset(BaseDataset):
    """A dataset class for paired image dataset that furthermore ingests segmented data from physics models.
    The physics model data comes in form of binary masks.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of different folders train_AB, test_AB and val_AB.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase,'_AB/').replace('/_', '_')  # get the image directory
        self.dir_A = os.path.join(opt.dataroot, opt.phase,'_A/').replace('/_', '_')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase,'_B/').replace('/_', '_')  # get the image directory
        print('=====',opt.phase, self.dir_AB, opt.dataroot_A_masks)
        # get the image physics super directory (train_mask, val_mask and test_mask folders must live here)
        # masks_folder_data_A = glob.glob(self.dir_A+"\\*mask.jpg") # mask
        if opt.phase == 'train': # ToDo check: is it needed to save the first 2?
            self.dir_AP = os.path.join(opt.dataroot_A_masks, '')#'train_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, '')#'train_mask')
            self.dir_P_pre = os.path.join(opt.dataroot_A_masks, '')#'train_mask')
            self.dir_P_post = os.path.join(opt.dataroot_B_masks, '')#'train_mask')
        elif opt.phase == 'test':
            self.dir_AP = os.path.join(opt.dataroot_A_masks, '')#'test_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, '')#'test_mask')
            self.dir_P_pre = os.path.join(opt.dataroot_A_masks, '')#'test_mask')
            self.dir_P_post = os.path.join(opt.dataroot_B_masks, '')#'test_mask')
        elif opt.phase == 'val':
            self.dir_AP = os.path.join(opt.dataroot_A_masks, '')#'val_mask')
            self.dir_BP = os.path.join(opt.dataroot_B_masks, '')#'val_mask')
            self.dir_P_pre = os.path.join(opt.dataroot_A_masks, '')#'val_mask')
            self.dir_P_post = os.path.join(opt.dataroot_B_masks, '')#'val_mask')
        else:
            print(" WRONG opt.phase option: ", opt.phase)
        print('dir_AB: {}, dir_A: {}, dir_B: {}, dir_P_pre: {}, dir_P_post: {}. [dir_AP could be the same as \
            dir_BP if no pre and post masks available (as it is as of now): {}\n{}'.format(self.dir_AB, self.dir_A, self.dir_B, self.dir_P_pre, self.dir_P_post, self.dir_AP, self.dir_BP))
        try:
            if not os.path.exists(self.dir_AP):
                raise Exception("Directory not accessible: {}. Train_mask, val_mask and test_mask folders must live here. \
                Please make sure you have downloaded both data and its segmentation masks. \n Example \
                Here xBD data and masks are in /home/jupyter/xBD.\n Download the data running in the terminal (placed in eie_vision): \n \
                sh scripts/downloads_gcp.sh".format(self.dir_AP))
            if not os.path.exists(self.dir_BP):
                raise Exception("Directory not accessible: {}. Train_mask, val_mask and test_mask folders must live here. \
                Please make sure you have downloaded both data and its segmentation masks. \n Example \
                Here xBD data and masks are in /home/jupyter/xBD.\n Download the data running in the terminal (placed in eie_vision): \n \
                sh scripts/downloads_gcp.sh".format(self.dir_BP))
        finally:
            print('Successfully read images and physics related segmentation masks ', self.dir_AB, self.dir_AP,self.dir_BP)

        # toDo meantime: (right now the segm. flood masks are the same for input and output images (A and B) and they correspond always to post flood images)
        # the 4th channel is the input condition to the conditional discriminator (default bicyclegan isnt conditional).
        # self.dir_AP = self.dir_AB # get the image physics super directory (train/, val/, test/)
        # self.dir_BP = self.dir_AB  # get the image physics super directory (train/, val/, test/)
        print('but while data is ready, it is for now: ', self.dir_AP, opt.max_dataset_size)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size)) # get image paths
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
        # self.AP_paths = sorted(make_dataset(self.dir_AP, opt.max_dataset_size))  # get image + mask paths
        # self.BP_paths = sorted(make_dataset(self.dir_BP, opt.max_dataset_size))  # get image + mask paths
        self.P_pre_paths = sorted(make_dataset(self.dir_P_pre, opt.max_dataset_size))  # get image mask paths
        self.P_post_paths = sorted(make_dataset(self.dir_P_post, opt.max_dataset_size))  # get image mask paths

        assert self.opt.load_size >= self.opt.crop_size, "crop_size should be smaller than the size of loaded image"
        # Extend to handle also Physics channel in the target image
        assert self.opt.direction == 'APtoBP' or self.opt.direction == 'BPtoAP', 'Valid options for this model are APtoBP or BPtoAP'
        assert len(self.P_pre_paths) == len(self.AB_paths), 'Number of images in training data --dataroot must be equal to the number of images \
            in the physics masks (--dataroot_A_masks and also in --dataroot_B_masks): AB: {}, P_pre: {}, P_post: {}'.format(len(self.AB_paths), len(self.P_pre_paths), len(self.P_post_paths))
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

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        # #if self.opt.direction == 'APtoBP':   #ToDo useles block??
        # AP_path = self.AP_paths[index]
        # #P_pre = Image.open(AP_path).convert('LA') #LA mode has luminosity (brightness) and alpha 
        # #else: #ToDo, later we can avoid creating either of them in each run as it is not running AtoB and BtoA always (but BzB and zBz).
        # BP_path = self.BP_paths[index]
        # #BP = Image.open(BP_path).convert('LA')

        P_pre_path = self.P_pre_paths[index]
        P_post_path = self.P_post_paths[index]

        P_pre = Image.open(P_pre_path).convert('LA') 
        P_post = Image.open(P_post_path).convert('LA') 
        

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        p_w, p_h = P_pre.size
        assert w2==p_w and h==p_h, 'Dimensions of masks do not match dimensions of input images, i.e. half of AB size: w: {}, h: {}. Physics masks P_pre: p_w: {}, p_h: {}.'.format(w2, h, p_w, p_h)
         
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=self.input_nc == 1)
        B_transform = get_transform(self.opt, transform_params, grayscale=self.output_nc == 1)
        #P_transform = get_transform(self.opt, transform_params, grayscale=True)  
        #convert = normalize      
        transform_physics_mask = get_transform(self.opt, transform_params, grayscale=True, convert=False, sloshed = True)
        
        A = A_transform(A)
        B = B_transform(B)
        # physics informing channel image with segmentation masks pre and post (flood) event 
        P_pre = transform_physics_mask(P_pre)
        P_post = transform_physics_mask(P_post)

        #print('P_post:{}'.format(P_post))
        # make the physics masks binary so they are more easy to visualize 
        # (as we apply normalize transform, the threshold should be 0)
        # P_pre = th.where(P_pre < 0.5, th.zeros_like(P_pre), th.ones_like(P_pre)) 
        # P_post = th.where(P_post < 0.5, th.zeros_like(P_post), th.ones_like(P_post)) 
        P_pre = th.where(P_pre < 0.5, -th.ones_like(P_pre), th.ones_like(P_pre)) 
        P_post = th.where(P_post < 0.5, -th.ones_like(P_post), th.ones_like(P_post)) 
        #print('after P_post:{}'.format(P_post))
        # merge A with the physical mask of A (AP) #print('AP before: ', P_pre.size(), P_post.size()) # (1,256,256)
        AP = th.cat((A, P_post), dim=0)
        BP = th.cat((B, P_pre), dim=0)
        
        return {'A': A, 'B': B, 'A_path': A_path, 'B_path': B_path, 'AP': AP, #'AP_path': AP_path, 
                'BP': BP, #'BP_path': BP_path, 
                'P_pre': P_pre, 'P_pre_path': P_pre_path, 'P_post': P_post, 'P_post_path': P_post_path}

    def random_split(self):
        # toDo
        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
