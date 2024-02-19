#!/bin/sh

# NOTE! Some of these variables are used in $rm -r statements. Do not leave variables empty or filled with '.'
bucket_path='there/is/no/bucket/path'
seg_model_path='pretrained/pix2pix_seg/houston_west/'
seg_model_name='latest_net_G.pth'

root_dir='/home/jupyter/eie_vision/'
dataroot='/data1/groups/EarthIntelligence/datasets/floods/processed/naip_to_all'
model_name='conditional_binary_naip_to_all/'
ckpt_dir='temp/checkpoint/Pix2pixHD/'
temp_dir='temp/Pix2pixHD/'
temp_mask='test_latest/mask/'
temp_ims='test_latest/images/'
results_dir='results/Pix2pixHD/'

# Baseline
im_baseline_dir='results/flood_color/conditional_binary_naip_to_all/'
split_prefix='test_'

im_suffix='_disaster'
suffix_pre='disaster.png'
suffix_post='disaster.png'
suffix_baseline='disaster.png'
suffix_mask='disaster.png'
event_name='houston_west_'
event='houston_west_'