# Earth Intelligence Engine
## Creating Physically-Consistent Visualizations of Climate Events with Deep Generative Vision Models

This is the official repository for the Earth Intelligence Engine. This code trains and evaluates a deep generative vision model (GAN) to synthesize physically-consistent imagery of future floods. The code also trains a flood segmentation model on aerial imagery. 

## Getting started

### Setup
```
git clone --recursive git@github.com:blutjens/earth-eie.git
cd earth-eie
conda env create -f conda.yaml
conda activate eie_vision
pip install -e .
```
We recommend setting up your environment with conda. If you're unfamiliar with conda, read [this intro](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307).

Why git clone `--recursive`? Because we have at least one git submodules for hosting models. This means **you'll need to run `git submodule update` when updating your remote.**

## Dataset
### Download from huggingface
Our full dataset, eie-earth-intelligence-engine, is available at huggingface. To download the dataset via git lfs please follow the instructions in the dataset [README.md](https://huggingface.co/datasets/blutjens/eie-earth-intelligence-engine)

## Reproduce the main results
### Train flood image-to-image (im2im) translation model
- For the main model follow the notebook at [link](sandbox/Pix2pixHD/Train_conditional_binary_scratch_spectral_lpips.ipynb). This notebook contains the terminal commands to train the flood im2im model on xbd2xbd. After training the model is used to create predictions over the test set and the flood segmentation model is used to create flood masks of the generated imagery.
- Monitor the training by opening [index.html](temp/checkpoint/Pix2pixHD/conditional_binary_spectral/web/index.html)

### Recreate the baseline flood visualization models
- The VAEGAN can be retrained with [link](sandbox/BicycleGAN/train_test_baseline.ipynb).
- The hand-generated baseline can be created with [link](sandbox/Color Baseline/Segment flood_color.ipynb).

### Evaluate im2im model
- Evaluate the imagery with eval_main() as called in [evaluate_notebook.ipynb](scripts/evaluate_notebook.ipynb)

## Optional: Reproduce auxiliary results
### Re-train the flood segmentation model on xbd-seg and create pre- and post-flood segmentations
- Train, evaluate the flood segmentation model by following our other repository [eie-flood-seg](https://github.com/blutjens/eie-flood-seg/blob/dev/sat2seg_crossval_scratch_1024_plus_houston_west.ipynb)
- Copy and paste the model weights from checkpoints/temp/ into pretrained/

### Train the generalization experiments for naip2xbd and naip2hou
- Follow the notebook [Train_conditional_binary_scratch_naip.ipynb](sandbox/Pix2pixHD/Train_conditional_binary_scratch_naip.ipynb)

### Extensions to forest, forest-gtm, and arctic imagery
- Train an Arctic sea ice segmentation model with [arctic-sea-ice-seg](https://github.com/blutjens/eie-flood-seg/blob/dev/sat2seg_crossval_scratch_1024_plus_arctic_sea_ice.ipynb)
- The code for generating reforestation visualizations is currently not available.

### Re-download and process the raw data
- xbd2xbd: Execute the steps in our [eie-preprocessing](https://github.com/blutjens/eie-preprocessing) repository to download and process the dataset. The first step will be to download the raw xBD flood imagery from xview by following the script at: eie-preprocessing/scripts/download_xBD_geotiles.sh
- xbd-seg: Hand-label data in xbd2xbd
- {naip2xbd, naip2hou, hou-seg}: Follow the instructions in the paper.
- arctic: Follow the instructions in full-pipeline/pipeline.sh in the [arctic-sea-ice](https://github.com/blutjens/eie-arctic-sea-ice/-/tree/final-pipeline-sentinel) repository
- {forest, forest-gtm}: Follow the instructions in the paper.

### Visualization 
- Visualize the generated imagery as a large geospatial map with [align_slosh_w_naip.ipynb](https://github.com/blutjens/eie-preprocessing/blob/master/scripts/align_slosh_w_naip.ipynb) -> "Create large tif from generated imagery"

## Folder structure
```
- archive: legacy code and documents
- configs: hyperparameters for the tested models
- data: placeholder for raw, interim, and processed data 
- docs: documentation, references, and figures
- pretrained: placeholder for model checkpoints
- results: generated imagery
- sandbox: prototyping scripts and notebooks
- scripts: important scripts and notebooks
- src: model source code, forked from existing git repositories
- temp: temporary results while training the models
```

## Reference
```
@misc{lutjens2024eie,
  author = {Lütjens, Björn and Leshchinskiy, Brandon and Boulais, Océane and Chishtie, Farrukh and Díaz-Rodríguez, Natalia and Masson-Forsythe, Margaux and Mata-Payerro, Ana and Requena-Mesa, Christian and Sankaranarayanan, Aruna and Piña, Aaron and Gal, Yarin and Raïssi, Chedy and Lavin, Alexander and Newman, Dava},
  title = {Satellite Imagery from the Future: Creating Physically-Consistent Visualizations of Climate Data with Deep Generative Vision Models},
  publisher = {in submission},
  year = {2024},
}
```
