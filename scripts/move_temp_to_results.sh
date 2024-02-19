#!/bin/bash

set -xe 

# Import paths (has to be called from root)
. scripts/config/houston_west.sh

# Move generated segmentation masks to results/ and clear masks from temp/ directory
src_dir=${temp_dir}${model_name}${temp_mask}
dst_dir=${results_dir}${model_name}'test_mask/'
cd ${root_dir}
rm -r ${dst_dir} || true # Remove pre-existing files in results dir
mkdir --parents ${dst_dir}
pwd
rm -r ${src_dir}${temp_ims}*real_A.png # || true # Remove unnecessary model outputs
rename 's/_fake_B.png/.png/' ${src_dir}${temp_ims}*
rename 's/_synthesized_image.png/.png/' ${src_dir}${temp_ims}*
mv ${src_dir}${temp_ims}* ${dst_dir}

# Move generated imagery to results/ and clear all model results from temp/ directory
src_dir=${temp_dir}${model_name}${temp_ims}
dst_dir=${results_dir}${model_name}'test_B/'

mkdir --parents ${root_dir}${dst_dir} 
rm -r ${root_dir}${dst_dir}* || true # Remove pre-existing files in results dir
rename 's/_synthesized_image.jpg/.jpg/' ${root_dir}${src_dir}
mv ${root_dir}${src_dir}* ${root_dir}${dst_dir}
rm -r ${temp_dir}${model_name} # Remove all model results from temp dir