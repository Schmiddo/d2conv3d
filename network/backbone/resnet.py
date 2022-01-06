import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from collections import OrderedDict

from utils.registry import register

from config.hack import get_global_config


try:
  from fast_depthwise_conv3d import DepthwiseConv3d
except ImportError as e:
  print("Could not load fast depthwise 3d conv")
  print(e)
  DepthwiseConv3d = nn.Conv3d


def _conv2d(inplanes, planes, kernel_size, stride=1, groups=1):
  return nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                   padding=int(kernel_size/2), groups=groups, bias=False)


def _conv3d(inplanes, planes, kernel_size, stride=1, groups=1, dilation=1):
  return nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                   padding=dilation*int(kernel_size/2), dilation=dilation,
                   groups=groups, bias=False)


class BasicBlock3d(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock3d, self).__init__()
    self.conv1 = _conv3d(inplanes, planes, kernel_size=3, stride=stride)
    self.bn1 = nn.BatchNorm3d(planes)
    self.conv2 = _conv3d(planes, planes, kernel_size=3, dilation=dilation)
    self.bn2 = nn.BatchNorm3d(planes)

    self.downsample = downsample

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = out.relu()

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      x = self.downsample(x)

    out += x
    out = out.relu()

    return out


class BottleneckBlock3d(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BottleneckBlock3d, self).__init__()
    self.conv1 = _conv3d(inplanes, planes, kernel_size=1)
    self.bn1 = nn.BatchNorm3d(planes)
    self.conv2 = _conv3d(planes, planes, kernel_size=3, stride=stride, dilation=dilation)
    self.bn2 = nn.BatchNorm3d(planes)
    self.conv3 = _conv3d(planes, planes * self.expansion, kernel_size=1)
    self.bn3 = nn.BatchNorm3d(planes * self.expansion)

    self.downsample = downsample

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = out.relu()

    out = self.conv2(out)
    out = self.bn2(out)
    out = out.relu()

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      x = self.downsample(x)

    out += x
    out = out.relu()

    return out


class BottleneckBlock3d_ip(BottleneckBlock3d):
  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BottleneckBlock3d_ip, self).__init__(inplanes, planes, stride, downsample, dilation)
    self.conv2 = nn.Sequential(
      _conv3d(planes, planes, kernel_size=1),
      nn.BatchNorm3d(planes),
      _conv3d(planes, planes, kernel_size=3, stride=stride,
              dilation=dilation, groups=planes)
    )


class BottleneckBlock3d_ir(BottleneckBlock3d):
  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BottleneckBlock3d_ir, self).__init__(inplanes, planes, stride, downsample, dilation)
    #self.conv2 = _conv3d(planes, planes, kernel_size=3, stride=stride,
    #                     dilation=dilation, groups=planes)
    self.conv2 = DepthwiseConv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation,
                                 dilation=dilation, groups=planes, bias=False)
    self.__checkpoint = get_global_config().checkpoint_backbone

  def forward(self, x):
    if x.requires_grad and self.__checkpoint:
      return checkpoint(super().forward, x)
    else:
      return super().forward(x)


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=400, strides=None, dilations=None):
    super(ResNet, self).__init__()
    self.inplanes = 64

    strides = strides or [1, 2, 2, 2]
    dilations = dilations or [1, 1, 1, 1]

    self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
    self.bn1 = nn.BatchNorm3d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], strides[0], dilations[0])
    self.layer2 = self._make_layer(block, 128, layers[1], strides[1], dilations[1])
    self.layer3 = self._make_layer(block, 256, layers[2], strides[2], dilations[2])
    self.layer4 = self._make_layer(block, 512, layers[3], strides[3], dilations[3])

    self.avgpool = nn.AdaptiveAvgPool3d(1)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    self.layer_widths = [64] + [w * block.expansion for w in (64, 128, 256, 512)]

  def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1):
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm3d(planes * block.expansion)
      )
    else:
      downsample = None

    layers = [block(self.inplanes, planes, stride, downsample, dilation)]
    self.inplanes = planes * block.expansion
    for i in range(1, num_blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool(self.relu(x))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)

    x = self.fc(x.flatten(1))

    return x

  def get_stem(self):
    return nn.Sequential(OrderedDict(
      conv1=self.conv1,
      bn1=self.bn1,
      relu=self.relu,
      maxpool=self.maxpool,
    ))


@register("backbone")
def resnet18(**kwargs):
  return ResNet(BasicBlock3d, [2, 2, 2, 2], **kwargs)


@register("backbone")
def resnet34(**kwargs):
  return ResNet(BasicBlock3d, [3, 4, 6, 3], **kwargs)


@register("backbone")
def resnet50(**kwargs):
  return ResNet(BottleneckBlock3d, [3, 4, 6, 3], **kwargs)


@register("backbone")
def resnet101(**kwargs):
  return ResNet(BottleneckBlock3d, [3, 4, 23, 3], **kwargs)


@register("backbone")
def resnet152(**kwargs):
  return ResNet(BottleneckBlock3d, [3, 8, 36, 3], **kwargs)


@register("backbone")
def resnet200(**kwargs):
  return ResNet(BottleneckBlock3d, [3, 24, 36, 3], **kwargs)


@register("backbone")
def resnet50_csn_ir(**kwargs):
  model = ResNet(BottleneckBlock3d_ir, [3, 4, 6, 3], **kwargs)
  model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                          padding=(1, 3, 3), bias=False)
  return model


@register("backbone")
def resnet152_csn_ip(**kwargs):
  model = ResNet(BottleneckBlock3d_ip, [3, 8, 36, 3], **kwargs)
  model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                          padding=(1, 3, 3), bias=False)
  return model


@register("backbone")
def resnet152_csn_ir(**kwargs):
  model = ResNet(BottleneckBlock3d_ir, [3, 8, 36, 3], **kwargs)
  model.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                          padding=(1, 3, 3), bias=False)
  return model
