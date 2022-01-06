# (Modulated) Deformable 3D Convolutions

This repository provides cuda implementations of 1D and 3D deformable [1] and modulated deformable [2] convolutions.

Use the `DConv` and `MDConv` modules as a drop-in replacement for regular convolutions:
```python
import torch
from dconv_native import DConv3d
from dconv_native.util import add_deformable_conv
from torchvision.models.video import r3d_18

model = r3d_18()
model.layer4, num_conversions = add_deformable_conv(model.layer4, version="DCNv1")
```
Or build your own deformable stuff:
```python
import torch
from dconv_native import deform_conv3d

N, C_in, C_out, D, H, W = 2, 6, 12, 8, 14, 14
groups, offset_groups = 2, 3
kernel_shape = 3, 3, 3

data = torch.randn(N, C_in, D, H, W).cuda()
weights = torch.randn(C_out, C_in//groups, *kernel_shape).cuda()

num_offsets = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * offset_groups
offsets = torch.randn(N, 3 * num_offsets, D, H, W).cuda()

# Deformable Convolution
out = deform_conv3d(data, offsets, None, weights, padding=(1, 1, 1),
 n_weight_groups=groups, n_offset_groups=offset_groups)

# Modulated Deformable Convolution
alpha = torch.randn(N, num_offsets, D, H, W).cuda()
out_modulated = deform_conv3d(data, offsets, alpha, weights, padding=(1, 1, 1),
 n_weight_groups=groups, n_offset_groups=offset_groups)
```

[1] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei. Deformable Convolutional Networks. ICCV 2017.
[Paper](http://arxiv.org/abs/1703.06211v3)

[2] Xizhou Zhu, Han Hu, Stephen Lin, Jifeng Dai. Deformable ConvNets v2: More Deformable, Better Results. CVPR 2019.
[Paper](https://arxiv.org/abs/1811.11168)

## Installation

### Requirements
Requires PyTorch >= 1.6, TorchVision >= 0.6, Cuda >= 10.2, GCC >= 8.
Might also work with earlier versions.
Set the environment variable `TORCH_CUDA_ARCH_LIST` to specify compute capabilities you want to compile for.
For example, to compile for a 1080ti (compute capability 6.1) and a Titan RTX (compute capability 7.5), you would use something like
```shell script
$ export TORCH_CUDA_ARCH_LIST="6.1;7.5"
```
before compiling.

### Python package
Install as a python package via 
```shell script
$ python setup.py install --user
```

## Testing

Run tests with
```shell script
$ python -m unittest test
```
If you encounter missing symbols when trying to run the tests, remove the package via pip, delete the folders created by `setup.py` and install again.

