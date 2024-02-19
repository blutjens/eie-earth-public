#!/bin/sh

# NOTE! Some of these variables are used in $rm -r statements. Do not leave variables empty or filled with '.'
bucket_path='eie-arctic-sea-ice/'
seg_model_path='pretrained/pix2pix_seg/arctic_sea_ice/'
seg_model_name='latest_net_G.pth'

root_dir='/home/jupyter/eie_vision/'
dataroot='data/eie-arctic-sea-ice-uniq/'
model_name='conditional_binary_scratch_arctic_sea_ice_uniq/'
ckpt_dir='temp/checkpoint/Pix2pixHD/'
temp_dir='temp/Pix2pixHD/'
temp_mask='test_latest/mask/'
temp_ims='test_latest/images/'
results_dir='results/Pix2pixHD/'

im_suffix='_melt'