#Adapted converter for caffe2 weights based on the work in https://github.com/moabitcoin/ig65m-pytorch

#!/usr/bin/env python3

import pickle
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, Conv2Plus1D, BasicBlock

import os.path as osp
from network.backbone.resnet import resnet50_csn_ir, resnet152_csn_ip, resnet152_csn_ir
from network.backbone.resnet import BottleneckBlock3d as Bottleneck


model_archs = {
    "resnet50_csn_ir": resnet50_csn_ir,
    "resnet152_csn_ir": resnet152_csn_ir,
    "resnet152_csn_ip": resnet152_csn_ip
}
model_blocks = {50: [0, 3, 7, 13, 16], 152: [0, 3, 11, 47, 50]}
g_blocks = []

def get_model(arch, num_classes=400):
    model = model_archs[arch](num_classes=num_classes)
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            print(
                "WARNING"
                " The channel-separated models use a BN epsilon of 1e-3,"
                " but pytorchs default is 1e-5. This parameter is not"
                " saved in the state dict, so make sure to set it"
                " in your code!"
            )
            m.eps = 1e-3
            # in the caffe2 source code momentum is 0.9, but it's used the other way around in pytorch, so we use 0.1
            # caffe2:  x_new = x_old * momentum + x * (1 - momentum)
            # pytorch: x_new = x_old * (1 - momentum) + x * momentum
            m.momentum = 0.1
    return model


def blobs_from_pkl(path, num_classes=400):
    with path.open(mode="rb") as f:
        pkl = pickle.load(f, encoding="latin1")
        blobs = pkl["blobs"]

        assert "last_out_L" + str(num_classes) + "_w" in blobs, \
            "Number of --classes argument doesnt matche the last linear layer in pkl"
        assert "last_out_L" + str(num_classes) + "_b" in blobs, \
            "Number of --classes argument doesnt matche the last linear layer in pkl"

        return blobs


def copy_tensor(data, blobs, name):
    tensor = torch.from_numpy(blobs[name])

    del blobs[name]  # enforce: use at most once

    assert data.size() == tensor.size(), f"Torch tensor has size {data.size()}, while Caffe2 tensor has size {tensor.size()}"
    assert data.dtype == tensor.dtype

    data.copy_(tensor)


def copy_conv(module, blobs, prefix):
    assert isinstance(module, nn.Conv3d)
    assert module.bias is None
    copy_tensor(module.weight.data, blobs, prefix + "_w")


def copy_bn(module, blobs, prefix):
    assert isinstance(module, nn.BatchNorm3d)
    copy_tensor(module.weight.data, blobs, prefix + "_s")
    copy_tensor(module.running_mean.data, blobs, prefix + "_rm")
    copy_tensor(module.running_var.data, blobs, prefix + "_riv")
    copy_tensor(module.bias.data, blobs, prefix + "_b")


def copy_fc(module, blobs):
    assert isinstance(module, nn.Linear)
    n = module.out_features
    copy_tensor(module.bias.data, blobs, "last_out_L" + str(n) + "_b")
    copy_tensor(module.weight.data, blobs, "last_out_L" + str(n) + "_w")


# https://github.com/facebookresearch/VMZ/blob/6c925c47b7d6545b64094a083f111258b37cbeca/lib/models/r3d_model.py#L233-L275
def copy_stem(module, blobs):
    assert isinstance(module, nn.Sequential)
    assert len(module) == 4
    copy_conv(module[0], blobs, "conv1_middle")
    copy_bn(module[1], blobs, "conv1_middle_spatbn_relu")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "conv1")


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_conv2plus1d(module, blobs, i, j):
    assert isinstance(module, Conv2Plus1D)
    assert len(module) == 4
    copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j) + "_middle")
    copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j) + "_middle")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "comp_" + str(i) + "_conv_" + str(j))


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_basicblock(module, blobs, i):
    assert isinstance(module, BasicBlock)

    assert len(module.conv1) == 3
    assert isinstance(module.conv1[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv1[0], blobs, i, 1)
    assert isinstance(module.conv1[1], nn.BatchNorm3d)
    copy_bn(module.conv1[1], blobs, "comp_" + str(i) + "_spatbn_" + str(1))
    assert isinstance(module.conv1[2], nn.ReLU)

    assert len(module.conv2) == 2
    assert isinstance(module.conv2[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv2[0], blobs, i, 2)
    assert isinstance(module.conv2[1], nn.BatchNorm3d)
    copy_bn(module.conv2[1], blobs, "comp_" + str(i) + "_spatbn_" + str(2))

    if module.downsample is not None:
        # TODO: adapt to chosen arch
        assert i in [3, 7, 13]
        assert len(module.downsample) == 2
        assert isinstance(module.downsample[0], nn.Conv3d)
        assert isinstance(module.downsample[1], nn.BatchNorm3d)
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")

def copy_bottleneck(module, blobs, i):
    assert isinstance(module, Bottleneck)

    copy_conv(module.conv1, blobs, "comp_" + str(i) + "_conv_" + str(1))
    copy_bn(module.bn1, blobs, "comp_" + str(i) + "_spatbn_" + str(1))

    #Adjust for a potential bug in naming of layers in Facebook net: second ID counts up
    #twice for a depthwise convolutional layer.
    copy_conv(module.conv2, blobs, "comp_" + str(i) + "_conv_" + str(3))
    copy_bn(module.bn2, blobs, "comp_" + str(i) + "_spatbn_" + str(3))

    copy_conv(module.conv3, blobs, "comp_" + str(i) + "_conv_" + str(4))
    copy_bn(module.bn3, blobs, "comp_" + str(i) + "_spatbn_" + str(4))

    if module.downsample is not None:
        assert i in g_blocks, str(i)
        assert len(module.downsample) == 2
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")


def copy_bottleneck_csn_ip(module, blobs, i):
    assert isinstance(module, Bottleneck)

    copy_conv(module.conv1, blobs, "comp_" + str(i) + "_conv_" + str(1))
    copy_bn(module.bn1, blobs, "comp_" + str(i) + "_spatbn_" + str(1))

    copy_conv(module.conv2, blobs, "comp_" + str(i) + "_conv_" + str(2) + '_middle')
    copy_bn(module.bn2, blobs, "comp_" + str(i) + "_spatbn_" + str(2)+ '_middle')

    copy_conv(module.conv3, blobs, "comp_" + str(i) + "_conv_" + str(2))
    copy_bn(module.bn3, blobs, "comp_" + str(i) + "_spatbn_" + str(2))

    copy_conv(module.conv4, blobs, "comp_" + str(i) + "_conv_" + str(3))
    copy_bn(module.bn4, blobs, "comp_" + str(i) + "_spatbn_" + str(3))

    if module.downsample is not None:
        assert i in g_blocks, str(i)
        assert len(module.downsample) == 2
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")


def init_canary(model):
    nan = float("nan")

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            nn.init.constant_(m.weight, nan)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.running_mean, nan)
            nn.init.constant_(m.running_var, nan)
            nn.init.constant_(m.bias, nan)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.bias, nan)


def check_canary(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            assert not torch.isnan(m.weight).any()
        elif isinstance(m, nn.BatchNorm3d):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.running_mean).any()
            assert not torch.isnan(m.running_var).any()
            assert not torch.isnan(m.bias).any()
        elif isinstance(m, nn.Linear):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.bias).any()


def main(args):
    global g_blocks
    blobs = blobs_from_pkl(args.pkl)

    model = get_model(args.model)

    init_canary(model)

    copy_conv(model.conv1, blobs, "conv1")
    copy_bn(model.bn1, blobs, "conv1_spatbn_relu")

    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    g_blocks = model_blocks[50 if "50" in args.model else 152]

    sum_blocks = 0
    for layer, i in zip(layers, g_blocks):
      assert sum_blocks == i
      sum_blocks += len(layer)

      j=i
      for bottleneck in layer:
        if("ip" in args.model):
          copy_bottleneck_csn_ip(bottleneck, blobs, j)
        else:
          copy_bottleneck(bottleneck, blobs, j)
        j += 1
    assert sum_blocks == g_blocks[-1]

    copy_fc(model.fc, blobs)

    assert not blobs
    check_canary(model)

    # Export to pytorch .pth and self-contained onnx .pb files

    torch.save(model.state_dict(), args.pkl.with_suffix(".pth"))

    # Check pth roundtrip into fresh model
    
    model = get_model(args.model)
    model.load_state_dict(torch.load(args.pkl.with_suffix(".pth")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("pkl", type=Path, help=".pkl file to read the weights from")
    arg("model", choices=model_archs.keys(), help="model type the weights belong to")

    main(parser.parse_args())
