#!/bin/bash
# A Simple Script To Sync results and xBD from Google Cloud
# Chris Requena-Mesa 29/07/2020

echo Authentify with gcloud
gcloud auth login --no-launch-browser

echo Download  train data in eie_xview_raw

#Attention be on the eie_vision directory!
mkdir xBD_full
mkdir xBD_full/train_A/
mkdir xBD_full/train_B/
mkdir xBD_full/train_mask/

gsutil -m cp -r gs://eie_xview_raw/training-set/train/images/* xBD_full/train_A
gsutil -m cp -r gs://eie_xview_raw/holdout-set/hold_images_labels_targets/hold/images/* xBD_full/train_A
gsutil -m cp -r gs://eie_xview_raw/additional-tier3-training-set/tier3/images/* xBD_full/train_A
mv xBD_full/train_A/*post_disaster.png xBD_full/train_B/

#create empty masks for non-flood disasters and copy used masks from xBD for the flood ones.
cd xBD_full
cp train_B/* train_mask/
cd train_mask/
for f in * ; do cp -r ../empty_mask.png "$f" ; done
cd ../..
mkdir xBD_full/train_flood_mask/
cp xBD/train_mask/* xBD_full/train_flood_mask/
cd xBD_full/train_flood_mask/

postf=".png"
for name in *
do
    newname="$(echo "$name" | cut -c27-)"
    newname="${newname%_fake_B.png}"
    mv "$name" "$newname$postf"
done

cd ..
cp train_flood_mask/* train_mask/
