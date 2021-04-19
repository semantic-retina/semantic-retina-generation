# Semantic Retina Synthesis

This repository contains code for retina synthesis using semantic labels.

## Dataset Preparation

Use the `preproprocess_dataset.py` script with paths to the IDRiD and FGADR datasets to
preprocess them into the form that the model expects. Then, use `split_datasets.py` to
create train/test splits from these. This will create CSV files in directory `data` that
is used by all models.

## SPADE

The image-to-image translation SPADE model is not included in this repository, but can be found on [GitHub](https://github.com/nvlabs/spade/).

## Training Models

Train the semantic label generation ACGAN with `train_acgan.py`. For example:
```shell
python train_acgan.py test_generation_model
```
Then, generate test samples with `test_acgan.py`, specifying the model name.
```shell
python test_acgan.py test_generation_model
```
Train the segmentation U-Net with `train_unet.py`.
```shell
python test_unet.py test_segmentation_model --n_synthetic 500
```
This can then be evaluated on the test set using `test_unet.py`.
```shell
python test_unet.py test_segmentation_model
```