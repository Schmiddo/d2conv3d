# D²Conv3D: Dynamic Dilated Convolutions for Object Segmentation in Videos
This repository contains the implementation for

**D²Conv3D: Dynamic Dilated Convolutions for Object Segmentation in Videos"** by Christian Schmidt, Ali Athar, Sabarinath Mahadevan, and Bastian Leibe.

WACV 2022 | [Paper](https://arxiv.org/abs/2111.07774)

## Setup
Setup the environment with `conda env create --file env.yaml`.
Then compile and install the cuda operators via
```shell
pushd ops/dconv-native
python setup.py install
popd

# For faster depthwise 3D convolutions (optional)
# Packages code from this pull request https://github.com/pytorch/pytorch/pull/51027
# probably not necessary beginning with PyTorch 1.9
pushd ops/fast-depthwise-conv3d
python setup.py install
popd
```

If conda installs torchvision 0.2.2, you can upgrade after the environment is set up with `pip install --upgrade torchvision` to the newest version.

If ffmpeg/libopenh264.so.5 are missing in your conda env, try:
```shell
cd <path-to-your-conda-env>/lib/
ln -s libopenh264.so libopenh264.so.5
```

You can set the environment variable `DATA` to point to the directory where DAVIS is located, or modify `config/paths/default.yaml`. These paths should be absolute.
See `config/defaults.py` for all configuration options.

By default, backbone weights are expected in `./saved_models/backbones`.
Download the backbone weights (ir-CSN-152, pretrained on Sports1M, finetuned on Kinetics) from [this site](https://github.com/facebookresearch/VMZ/blob/main/c2/tutorials/model_zoo.md) and put it in the backone folder.


## Running the code
Example run scripts are provided in `run_scripts`.
Set the environment variable `SOURCE_DIR` to the path to this repository.
The scripts use as many GPUs as are available; we ran our experiments on two V100.
To train, simply run one of these scripts:
```shell
bash run_scripts/xyz.sh
```
To test after training, run
```shell
bash run_scripts/xyz.sh test
# With a certain checkpoint
WEIGHTS=<path-to-checkpoint> bash run_scripts/xyz.sh test
```
By default, the results are saved in `runs/<run>/results_5/vos/` with a temporal gap of 5 and `runs/<run>/results_1/vos/` for dense evaluation.
To compute the J&F scores, use the [DAVIS evaluation package](https://github.com/davisvideochallenge/davis2017-evaluation) adapted for DAVIS-16 as described [here](https://github.com/davisvideochallenge/davis2017-evaluation/issues/4).

