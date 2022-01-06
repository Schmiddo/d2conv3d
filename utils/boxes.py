import torch
import einops


def draw_box(frame, box):
  l, t, r, b = box
  frame[t, l:r] = 1
  frame[b, l:r] = 1
  frame[b:t, l] = 1
  frame[b:t, r] = 1


def segmap_to_boxes(mask, ids=None):
  boxes = []
  ids = mask.unique() if ids is None else ids
  for id in ids:
    if id == 0: continue
    inst_mask = torch.nonzero(mask == id)
    if inst_mask.numel() > 0:
      l = inst_mask[:, 1].min()
      r = inst_mask[:, 1].max()
      t = inst_mask[:, 0].max()
      b = inst_mask[:, 0].min()
      boxes.append(torch.as_tensor([l, t, r, b]))
  return boxes


def single_segmap_to_boxmap(mask):
  assert mask.ndim == 2
  boxmap = torch.zeros(*mask.shape, 4, device=mask.device)
  ids = mask.unique()
  for id in ids:
    if id == 0: continue
    inst_mask = torch.nonzero(mask == id)
    l = inst_mask[:, 1].min()
    r = inst_mask[:, 1].max()
    t = inst_mask[:, 0].min()
    b = inst_mask[:, 0].max()
    boxmap[inst_mask.unbind(1)] = torch.as_tensor([l, t, r, b], dtype=torch.float, device=mask.device)
  return boxmap


def batched_segmap_to_boxmap(mask):
  assert mask.ndim == 4
  N = mask.shape[0]
  mask = einops.rearrange(mask, "b d h w -> (b d) h w")
  boxmap = torch.stack([single_segmap_to_boxmap(m) for m in mask])
  return einops.rearrange(boxmap, "(b d) h w c -> b d h w c", b=N)


def clip_to_boxmap(mask):
  return torch.stack([segmap_to_boxmap(frame) for frame in mask.unbind(0)])
