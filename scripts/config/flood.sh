#!/bin/sh

# NOTE! Some of these variables are used in $rm -r statements. Do not leave variables empty or filled with '.'
bucket_path='eie_floods/naip/xbd/images/'
seg_model_path='pretrained/pix2pix_seg/flood/scratch_1024_plus/'
seg_model_name='latest_net_G.pth'

root_dir='/home/jupyter/eie_vision/'
dataroot='data/eie_floods/naip/xbd/images/'
model_name='conditional_binary/'
ckpt_dir='temp/checkpoint/Pix2pixHD/'
temp_dir='temp/Pix2pixHD/'
temp_mask='test_latest/mask/'
temp_ims='test_latest/images/'
results_dir='results/Pix2pixHD/'

# Baseline
im_baseline_dir='results/flood_color/conditional_binary'
split_prefix="test-set_test_images_labels_targets_test_images_"

im_suffix='_disaster'
suffix_pre='pre_disaster.png'
suffix_post='post_disaster.png'
suffix_baseline='post_disaster.png'
suffix_mask='post_disaster.png'
event_name='hurricane-'
event='hurricane-harvey_'