import torch
import einops
import os
from imageio import imwrite, mimwrite
from collections import defaultdict, abc
from itertools import repeat

import time


# Helper from torch's ConvNd
def _ntuple(n):
  def parse(x):
    if isinstance(x, abc.Iterable):
      return x
    return tuple(repeat(x, n))

  return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Timer:
  def __init__(self, name="Snippet", string="{name} took {time:5f}s", synchronize=False, output=True):
    self.start = None
    self.name = name
    self.string = string
    self.synchronize = synchronize
    self.output = output
    self.laps = []

  def mean(self):
    return sum(self.laps)/len(self.laps)

  def total(self):
    return sum(self.laps)

  def laps(self):
    return len(self.laps)

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.synchronize:
      torch.cuda.synchronize()
    self.laps.append(time.time() - self.start)
    if self.output:
      print(self.string.format(name=self.name, time=self.laps[-1], mean=sum(self.laps)/len(self.laps)))


def save_video(save_path, video, format=None):
  if format in ("jpg", "png"):
    os.makedirs(save_path, exist_ok=True)
    if len(video.shape) == 4:
      assert video.shape[0] in (1, 3, 4)
      video = video.permute(1, 2, 3, 0).squeeze(-1)
    if format == "png":
      video = video.to(torch.uint8)
    num_frames = video.shape[0]
    for f in range(num_frames):
      imwrite(os.path.join(save_path, f"{f:05}.{format}"), video[f].numpy())
  else:
    mimwrite(save_path, video, format)


def find_most_recent_checkpoint(checkpoint_dir):
  if not os.path.isdir(checkpoint_dir):
    return None

  checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

  if checkpoints:
    return os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda c: int(c[:-5]))[-1])
  else:
    return None


def load_weights(model: torch.nn.Module, weights_path, ignore_keys=None, strict=True, subkey=None):
  state_dict = torch.load(weights_path, map_location="cpu")
  if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
  elif "model" in state_dict:
    state_dict = state_dict["model"]
  if not hasattr(model, "module"):
    # in case the checkpoint was saved from DDP training, but model is not DDP
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  if subkey:
    state_dict = {k[len(subkey):]: v for k, v in state_dict.items()}

  ignore_keys = ignore_keys or []
  for key in ignore_keys:
    if key in state_dict: del state_dict[key]
  status = model.load_state_dict(state_dict, strict)
  if len(status.unexpected_keys) > 0:
    raise RuntimeError(
      f"Unexpected keys for {model.__class__.__name__}:"
      f"{status.unexpected_keys}"
    )

  del state_dict
  #torch.cuda.empty_cache()


def partition(predicate, module):
  f = lambda n, p: int(not predicate(n, p))
  return partition_n(f, module, 2)


def partition_n(fn, module, n):
  out = [defaultdict(dict) for _ in range(n)]
  for param_name, param in module.named_parameters():
    i = fn(param_name, param)
    assert 0 <= i < n, f"{i} must be in range [0, {n})"
    out[i][param_name] = param
  return tuple(o for o in out)


class Font:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

  @classmethod
  def bold(cls, text):
    return cls.BOLD + str(text) + cls.END

  @classmethod
  def red(cls, text):
    return cls.RED + str(text) + cls.END


def fill_segmentation_map(mask, cmap):
  """
  Fills a mask of class labels according to the given color map.
  Output is of the same type as the color map.

  :param mask: segmentation mask. Must be of type torch.long and of
      shape NxDxHxW
  :param mapping: must be num_classes x c
  :param ids: when given, colors only these classes; all entries must be < num_classes
  :return: filled segmentation map, shape NxDxHxWxc
  """
  shape = mask.shape
  d = cmap.shape[1]
  indices = mask.long()[..., None].expand(*shape, d).view(mask.numel(), d)
  return torch.gather(cmap, 0, indices).view(*shape, d).to(cmap.dtype)


def color_map(N=256, normalized=False):
  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  cmap = torch.zeros(N, 3, dtype=torch.float if normalized else torch.uint8)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = torch.as_tensor([r, g, b])

  cmap = cmap / 255 if normalized else cmap
  return cmap
