import torch
import torch.nn.functional as F
import einops

from torch.nn.modules.utils import _triple


def grid_like(tensor):
  """
  Constructs a regular grid with the same spatial dimensions as the input tensor,
  on the same device. For an input of shape NxCxD_1x...xD_n returns grid of shape
  nxD_1x...xD_n, i.e. the size of the first dimension is equal to the number of
  spatial dimensions.

  :param tensor: tensor of reference shape/device
  :return: A regular grid
  """
  return torch.stack(torch.meshgrid([torch.arange(0, d, device=tensor.device) for d in tensor.shape[2:]]))


def get_kernel_grid(kernel_shape, dilation, device="cpu"):
  return torch.stack(torch.meshgrid([
    torch.linspace(-d * (k//2), d * (k//2), k, device=device)
    for d, k in zip(dilation, kernel_shape)
  ]))


def get_convolution_grid(input_shape, kernel_shape, stride, padding, dilation, device="cpu"):
  return torch.stack(torch.meshgrid([
    torch.arange(a * (k//2) - p, d - a * (k//2) + p, s, device=device)
    for d, k, s, p, a in zip(input_shape, kernel_shape, stride, padding, dilation)
  ]))


@torch.no_grad()
def construct_3d_kernel_grid(input_size, kernel_size=3, stride=1, pad=1, dilation=1, device="cpu"):
  """
  Given parameters for a 3d convolution, this method computes the coordinates
  of the input where a filter point is multiplied.
  Output is (K*K*K)x3xDxHxW.
  All inputs should be triples.
  """

  kernel_size = _triple(kernel_size)
  stride = _triple(stride)
  pad = _triple(pad)
  dilation = _triple(dilation)

  # coordinates of each kernel point, relative to kernel's center
  # 3xKxKxK
  kernel_grid = get_kernel_grid(kernel_size, dilation, device)

  # coordinates of each kernel application (center point)
  # 3xDxHxW
  position_grid = get_convolution_grid(input_size, kernel_size, stride, pad, dilation, device)

  kernel_grid = einops.rearrange(kernel_grid, "c k1 k2 k3 -> (k1 k2 k3) c 1 1 1", c=3)
  grid = kernel_grid + position_grid
  return grid


def offsets_from_size_map(size_map, kernel_shape=(3, 3, 3), dilation=(1, 1, 1)):
  base_offset = get_kernel_grid(kernel_shape, dilation, size_map.device)
  base_offset = einops.rearrange(base_offset, "zyx kz ky kx -> kz ky kx zyx 1 1 1", zyx=3)

  # inflate to Nx3xDxHxW
  ones = torch.ones_like(size_map[:, 0]).unsqueeze(1)
  if size_map.shape[1] == 3:
    # no-op, size_map is already 3-dimensional
    pass
  elif size_map.shape[1] == 2:
    size_map = torch.cat([ones, size_map], dim=1)
  elif size_map.shape[1] == 1:
    size_map = torch.cat([ones, size_map, size_map], dim=1)
  else:
    raise ValueError(f"size_map should have 1, 2, or 3 channels, but has {size_map.shape[1]}")
  size_map = einops.rearrange(size_map, "n zyx d h w -> n 1 1 1 zyx d h w", zyx=3)

  offsets = base_offset * (size_map - 1)
  return einops.rearrange(offsets, "n kz ky kx zyx d h w -> n (kz ky kx zyx) d h w")


def generate_offsets(size_map, flow_map=None, kernel_shape=(3, 3, 3), dilation=(1, 1, 1)):
  """
  Generates offsets for deformable convolutions from scalar maps.
  Maps should be of shape NxCxDxHxW, i.e. one set of parameters for every
  pixel. ``size_map`` and ``orientation_map`` expect a single channel,
  and flow_map should have 2
  """

  kernel_shape = _triple(kernel_shape)
  dilation = _triple(dilation)

  zeros = torch.zeros_like(size_map)
  if flow_map is None:
    flow_map = torch.cat([zeros, zeros], dim=1)

  """
  at each pixel, size_map predicts the size of the object at that pixel
  0. Inverse optical flow if not given
  1. Sample sizes from previous/next frame according to optical flow
  2. scale, and shift kernel shape
    a. Multiply with sizes
    b. Add optical flow shift
  """
  if flow_map.shape[1] == 2:
    flow_prev = flow_map
    # TODO: properly inverse optical flow
    flow_next = -flow_map
  else:
    flow_prev, flow_next = torch.split(flow_map, [2, 2], dim=1)

  kz, ky, kx = kernel_shape
  assert kz == ky == kx == 3, "not implemented for kernel shapes != 3x3x3"
  N, _, D, H, W = size_map.shape

  grid = grid_like(size_map)
  grid_prev = einops.rearrange(grid + flow_prev, "n zyx d h w -> n d h w zyx")
  grid_next = einops.rearrange(grid + flow_next, "n zyx d h w -> n d h w zyx")

  size_prev = F.grid_sample(size_map, grid_prev, align_corners=False)
  size_next = F.grid_sample(size_map, grid_next, align_corners=False)

  base_offset = get_kernel_grid(kernel_shape, dilation, size_map.device)
  base_offset = einops.rearrange(base_offset, "zyx kz ky kx -> kz ky kx zyx 1 1 1", zyx=3)

  def _inflate(m):
    return torch.cat([zeros, m, m], dim=1)

  offsets = torch.cat([
    base_offset[i] * _inflate(size)[:, None, None, None, ...]
    for i, size in enumerate((size_prev, size_map, size_next))
  ], dim=1)
  offsets = offsets + flow_map

  return offsets - base_offset

