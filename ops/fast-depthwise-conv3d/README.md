# Fast depthwise 3D convolutions
This repository provides a drop-in replacement for pytorch's `nn.Conv3d` when the number of input channels is the same as the number of groups.
The implementation is taken from the  corresponding [pull-request](https://github.com/pytorch/pytorch/pull/51027) in the pytorch repo.
The goal is to make the faster kernels available for older + stable pytorch versions.

The kernels are roughly 3-10 times faster than pytorch's and cudnn's regular conv kernels.

## Installation
```shell script
python setup.py install
```

Test with
```shell script
python -m unittest test
```

## Usage
Use it like a regular `nn.Conv3d`:
```python
from fast_depthwise_conv3d import DepthwiseConv3d

# use it like a regular nn.Conv3d; the 'groups' argument is ignored,
# internally it is always groups=in_channels
conv = DepthwiseConv3d(2, 4, kernel_size=3, groups=2)
```
Save memory _and_ compute by checkpointing.
For example:
```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from fast_depthwise_conv3d import DepthwiseConv3d

class BottleneckBlock_ir(nn.Sequential):
  def __init__(self, c_in, c_out):
    c_m = c_in//4
    super(BottleneckBlock_ir, self).__init__(
      nn.Conv3d(c_in, c_m, 1, bias=False),
      nn.BatchNorm3d(c_m),
      nn.ReLU(),
      DepthwiseConv3d(c_m, c_m, 3, padding=1, bias=False),
      nn.BatchNorm3d(c_m),
      nn.ReLU(),
      nn.Conv3d(c_m, c_out, 1, bias=False),
      nn.BatchNorm3d(c_out),
      nn.ReLU(),
    )

  def forward(self, x):
    return checkpoint(super().forward, x)
```
