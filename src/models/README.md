We support external models as git submodules, exposed via the `MLProjectBase` API in base_interface.py.

All models must be reflected in the dunder-init file in order to be found by the repo's main experiment runner.


We are using BycicleGAN and Pix2pixHD as our main models for the translation of pre- to post-flood images.
We use pix2pix for the task of segmenting the flood extent from RGB (for conditioning and evaluation)
We use CycleGAN for the domain adaptation task, i.e, transforming segmentation masks into SLOSH-like tiles.