import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from utils.registry import register

from utils.grid import construct_3d_kernel_grid


class StableBCELoss(torch.nn.modules.Module):
  def __init__(self):
    super(StableBCELoss, self).__init__()

  def forward(self, input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def _avg_pool(mask, shape):
  return einops.reduce(
    mask, "n c (d1 d2) (h1 h2) (w1 w2) -> n c d1 h1 w1",
    "mean", d1=shape[-3], h1=shape[-2], w1=shape[-1]
  )


def _interpolate(mask, shape):
  return F.interpolate(mask, size=shape, mode="nearest")


def _interpolate_bilinear(mask, shape):
  mask = F.interpolate(mask, size=shape, mode="trilinear", align_corners=False)
  mask = torch.where(mask > 0.1, 1, 0)
  return mask


# TODO: move hook from loss.util here
@register("loss", "saliency")
class ModulationLoss(nn.Module):
  """
  Apply binary cross entropy loss between modulation values and fg/bg masks

  Downscaling of gt masks either via avg pooling or interpolation
  """
  def __init__(self, weight=1.0, mode="interpolate"):
    super(ModulationLoss, self).__init__()
    if mode not in ("pool", "interpolate", "interpolate_bilinear"):
      raise ValueError(f"Expected scaling to be 'pool' or 'interpolate', but was {mode}")
    if mode == "pool":
      self.mode = mode
      self.loss_fn = nn.MSELoss()
      self.scale_fn = _avg_pool
    else:
      self.mode = mode
      self.loss_fn = nn.BCEWithLogitsLoss()
      self.scale_fn = _interpolate
    self.weight = weight

  def forward(self, deform_maps, gt_mask):
    offset_maps = {n: m[0] for n, m in deform_maps.items()}
    modulation_maps = {n: m[1] for n, m in deform_maps.items()}
    if self.mode == "pool":
      modulation_maps = {n: m.sigmoid() for n, m in modulation_maps.items()}

    # what shapes/resolutions do we have?
    # map shape is BxKxDxHxW, K is total number of kernel points (i.e. 27 for 3x3x3)
    shapes = {k: v.shape[-4:] for k, v in modulation_maps.items()}
    distinct_shapes = set(shapes.values())

    # resize gt maps to resolutions of corresponding offset/modulation maps
    # pad to get linear interpolation for out-of-bounds samples
    gt_mask = gt_mask.unsqueeze(1).float()
    gt_masks = {
      shape: F.pad(
        self.scale_fn(gt_mask, shape[-3:]),
        (1,1,1,1,1,1)
      )
      for shape in distinct_shapes
    }

    # sample gt masks according to offsets
    # TODO: support other kernel shapes than 3x3x3
    grids = {
      shape: construct_3d_kernel_grid(shape[-3:], device=gt_mask.device)
      for shape in distinct_shapes
    }

    tensor_shapes = {
      k: torch.as_tensor(v[-3:], device=gt_mask.device)
      for k, v in shapes.items()
    }
    saliency_loss = gt_mask.new_zeros(1)
    for n, m in modulation_maps.items():
      with torch.no_grad():
        offset_map = einops.rearrange(
          offset_maps[n],
          "n (K c) d h w -> n K c d h w", c=3
        )
        sample_grid = grids[shapes[n]] + offset_map
        sample_grid = einops.rearrange(
          sample_grid,
          "n K c d h w -> n (K d) h w c"
        )
        # pad to get linear interpolation for out-of-bounds
        sample_grid = 2 * ((1+sample_grid) / (tensor_shapes[n]+2)) - 1

        sampled_gt_mask = F.grid_sample(
          gt_masks[shapes[n]],
          sample_grid,
          mode="bilinear",
          align_corners=True
        )
        sampled_gt_mask = einops.rearrange(
          sampled_gt_mask,
          "n 1 (K d) h w -> n K d h w", K=shapes[n][0]
        )
        #print(sampled_gt_mask.sum(), gt_masks[shapes[n]].sum())
      saliency_loss += self.loss_fn(m, sampled_gt_mask)
    if len(modulation_maps) > 0:
      saliency_loss = saliency_loss / len(modulation_maps)
    return self.weight * saliency_loss.squeeze()
