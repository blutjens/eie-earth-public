README


Physically informed BicycleGAN



## Requirements:

pip install tensorboardX
pip install mlflow

or with conda
conda activate base
conda install -c conda-forge tensorboardx



# Steps to run

1 Download data:
xBD:
If xBD was previously download, create only the aligned datasets by running:

/home/jupyter$ python create_aligned_dataset_AB.py 

Otherwise, download xBD dataset running the download_xBD notebook:

https://github.com/NataliaDiaz/PhysicsInformedBicycleGAN/blob/master/data/download_xBD_dataset_keeping_orig_names.ipynb

SLOSH Dataset
cd /home/jupyter;  gsutil -m cp -r gs://eie_xview_processed/ ./eie_xview_processed/

2 Download masks
```
~/xBD/bigtiles$ gsutil -m cp -r gs://eie_floods/masks_v1 .
```


3 Example of training run: First time, --continue_train should not be used. If training breaks, or to continue training, you can use the flag `--continue_train` and `--epoch_count 126`. You keep the old flags the same: i.e., `--niter 100 --niter_decay 100` if you want a total of 200 epochs training. 

Example (working 24jul en bicycle-gan-tesla VM):

```
python ./BicycleGAN/train.py --model physics_informed_bicycle_gan --name xBD_BicycleGAN_256_physics_informed_v02_24jul --gpu_ids 0,1,2,3 --direction APtoBP --dataset_mode physics_aligned --dataroot ./xBD/bigtiles/ --niter 1 --niter_decay 0 --save_epoch_freq 1 --display_id -1 --load_size 1024 --crop_size 256 --batch_size 2  --conditional_D --dataroot_A_masks ./xBD/bigtiles/masks_v1  --dataroot_B_masks ./xBD/bigtiles/masks_v1
```

Remember to specify these flags after first run: 
```'--continue_train', action='store_true', help='continue training: load the latest model')
'--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
```       

For 1024 images so they are comparable to pix2pixHD pipeline:

```
/home/jupyter$ python ./BicycleGAN/train.py --model physics_informed_bicycle_gan --name xBD_BicycleGAN_1024_physics_informed_v00_24jul --gpu_ids 0,1,2,3 --direction APtoBP --dataset_mode physics_aligned --dataroot ./xBD/bigtiles/ --niter 100 --niter_decay 100 --save_epoch_freq 1 --display_id -1 --load_size 1024 --crop_size 1024 --batch_size 8  --conditional_D --dataroot_A_masks ./xBD/bigtiles/masks_v1  --dataroot_B_masks ./xBD/bigtiles/masks_v1 --netD basic_1024_multi --netD2 basic_1024_multi --netG unet_1024 --netE resnet_1024
```


Testing the model:

```
python ./PhysicsInformedBicycleGAN/test.py --dataroot ./xBD/bigtiles/ --dataset_mode physics_aligned --results_dir ./results/  --checkpoints_dir ./checkpoints/  --name testphysicsaligned --gpu_ids 0,1,2,3 --direction APtoBP  --load_size 1024  --crop_size 1024 --input_nc 4 --output_nc 4 --n_samples 1 --center_crop --no_flip --conditional_D
```


