# 6.8301 Final Project
### Harsh Deep, Alex Hu, Rithvik Ganesh

## Installation

Install poetry, then run `poetry install`

## Dataset:
Run `make data` to download the dataset, there is an iterable dataset in `src/data/dataset.py`

## Model
Model modified from [here](https://github.com/milesial/Pytorch-UNet/tree/master)

## Training
Prepare data in `data/raw/train`, `data/raw/val`
`python train.py --patch --run_dir runs/1`

## Report
See our report [here](https://drive.google.com/file/d/1Uu3Se25D98e8lFfb-NNZxB-xkJPGdQOR/view?usp=sharing).
