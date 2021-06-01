# SuperGAN Dataset Preprocessing

## About
This repository contains the scripts for preprocessing the datasets used in the SuperGAN paper. It does not, however, contain any of the datasets, preprocessed or otherwise. These datasets can be found at the following locations:

[Daily and Sports Activities Dataset](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities)
[CASAS ADL Activities Dataset](http://casas.wsu.edu/datasets/adlnormal.zip)

## Usage
Download the datasets in question and extract to the following folders:
* `sportsdata` for the Daily and Sports Activities Dataset
* `adlnormal` for the CASAS ADL Activities Dataset

Then run each preprocess script to yield an .h5 file containing the preprocessed dataset. These scripts will require the following Python3 packages:
* h5py
* numpy