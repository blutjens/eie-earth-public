import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_augmentation
from data.image_folder import make_dataset
from PIL import Image
from torch import cat
import torch
import torchvision.transforms as transforms
import random

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AlignedPhysicsBinDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A'
        dir_mask = '_mask' 
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.dir_mask = os.path.join(opt.dataroot, opt.phase + dir_mask)
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.mask_paths = sorted(make_dataset(self.dir_mask))
        
        if opt.isTrain:
            random.Random(4).shuffle(self.A_paths)
            random.Random(4).shuffle(self.mask_paths)
            
        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            if opt.isTrain:
                random.Random(4).shuffle(self.B_paths)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        mask_path = self.mask_paths[index]   
        A = Image.open(A_path)
        Mask = Image.open(mask_path)                             
        params = get_params(self.opt, A.size)
        
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            transform_mask = get_transform(self.opt, params, grayscale=True,normalize=False)
            if self.opt.augmentation_aggressive:
                augmentation_A = get_augmentation(fill=0)
                augmentation_mask = get_augmentation(fill=255)
                transform_A = transforms.Compose([augmentation_A, transform_A])
                transform_mask = transforms.Compose([augmentation_mask, transform_mask])
            
            A_tensor = transform_A(A.convert('RGB'))
            mask_tensor = transform_mask(Mask.convert('RGB'))
            mask_tensor = torch.where(mask_tensor < 0.5, -torch.ones_like(mask_tensor), torch.ones_like(mask_tensor)) #Make mask binary!
            all_tensor = cat((mask_tensor,A_tensor), dim=0)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            if self.opt.augmentation_aggressive:
                augmentation_B = get_augmentation(fill=0)
                transform_B = transforms.Compose([augmentation_B, transform_B])
            B_tensor = transform_B(B)
                                  
        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'mask': mask_tensor,'label': all_tensor, 'image_A': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedPhysicsBinDataset'