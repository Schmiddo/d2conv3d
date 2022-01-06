import torch
from abc import abstractmethod

import numpy as np
from imageio import imread

from torch.utils.data import Dataset

from utils.resize import resize, ResizeMode

from .util import flip_video_sample, crop_video_sample


TARGETS = 'targets'
IMAGES_ = 'images'
INFO = 'info'


class BaseDataset(Dataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None):
    self.resize_mode = ResizeMode(resize_mode)
    self.resize_shape = resize_shape
    self.mode = mode
    self.root = root
    self.raw_samples = []
    self.samples = []
    self.create_sample_list()

  def filter_samples(self, filter):
    """
    Filter dataset with predicate.

    :param filter: maps sample to true or false.
    :return: None
    """
    self.samples = [s for s in self.raw_samples if filter(s)]


  # Override in case tensors have to be normalised
  def normalise(self, sample):
    sample[IMAGES_] = torch.from_numpy(sample[IMAGES_]).float() / 255.0
    return sample

  def is_train(self):
    return self.mode == "train"

  def _get_spatial_dim(self, t):
    return t.shape[:2]

  def _get_spatial_padding(self, t):
    h, w = self._get_spatial_dim(t)
    new_h = h + 32 - h % 32 if h % 32 > 0 else h
    new_w = w + 32 - w % 32 if w % 32 > 0 else w
    lh, uh = (new_h - h) / 2, (new_h - h) / 2 + (new_h - h) % 2
    lw, uw = (new_w - w) / 2, (new_w - w) / 2 + (new_w - w) % 2
    return [(int(lh), int(uh)), (int(lw), int(uw))]

  def _get_full_padding(self, t, padding):
    return padding + [(0, 0)] if len(t.shape) == 3 else padding

  def _spatial_pad(self, t, padding):
    return np.pad(t, self._get_full_padding(t, padding))

  def pad_tensors(self, sample):
    padding = self._get_spatial_padding(sample[IMAGES_])
    sample[IMAGES_] = self._spatial_pad(sample[IMAGES_], padding)
    targets = sample[TARGETS]
    for k, v in targets.items():
      if k == "boxes":
        continue
      elif isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) >= 2:
        targets[k] = self._spatial_pad(v, padding)

    sample[INFO]["pad"] = padding
    return sample

  def read_sample(self, sample):
    sample[IMAGES_] = self.read_image(sample)
    sample[INFO]["shape"] = self._get_spatial_dim(sample[IMAGES_])
    sample[TARGETS] = self.read_target(sample)
    return sample

  @abstractmethod
  def read_target(self, sample):
    pass

  @abstractmethod
  def read_image(self, sample):
    pass

  def resize_sample(self, sample):
    data = {IMAGES_: sample[IMAGES_]}
    targets = sample[TARGETS]
    for k, v in targets.items():
      if k == "boxes":
        continue
      elif isinstance(v, (np.ndarray, torch.Tensor)) and len(v.shape) >= 2:
        data[k] = v

    data = resize(data, self.resize_mode, self.resize_shape)

    sample[IMAGES_] = data.pop(IMAGES_)
    for k in data:
      sample[TARGETS][k] = data[k]

    return sample

  def apply_augmentations(self, sample):
    #sample = flip_video_sample(sample, 0.5, 0.0, 0.0)
    #sample = crop_video_sample(sample, (0.8, 1.0))
    #sample["images"] = sample["images"][...]
    return sample

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    sample = self.read_sample(self.samples[idx].copy())

    if self.is_train():
      sample = self.apply_augmentations(sample)

    resized_sample = self.resize_sample(sample)

    padded_sample = self.pad_tensors(resized_sample)

    normalised_sample = self.normalise(padded_sample)

    return normalised_sample

  @abstractmethod
  def create_sample_list(self):
    pass


# TODO: Fix clip creation in VideoClips based implementations
# VideoClips creates clips in regular intervals but may omit frames
# at the end of videos
class VideoDataset(BaseDataset):
  def __init__(self, root, mode='train', resize_mode=None, resize_shape=None,
               tw=8, max_temporal_gap=8):
    self.tw = tw
    self.max_temporal_gap = max_temporal_gap

    self.videos = []
    self.raw_samples = []
    self.current_video = None

    super(VideoDataset, self).__init__(root, mode, resize_mode, resize_shape)

  def _get_spatial_dim(self, t):
    return t.shape[1:3]

  def _get_full_padding(self, t, padding):
    padding = [(0, 0)] + padding
    return padding + [(0, 0)] if len(t.shape) == 4 else padding

  def read_image(self, sample):
    images = map(lambda x: imread(x, pilmode="RGB"), sample['images'])
    return np.stack([i for i in images])

  def normalise(self, sample):
    sample = super(VideoDataset, self).normalise(sample)
    sample[IMAGES_] = sample[IMAGES_].permute(3, 0, 1, 2)
    return sample


class VideoSegmentationDataset(VideoDataset):
  def __init__(self, root, mode="train", resize_mode=None, resize_shape=None,
               tw=8, max_temporal_gap=8):
    self.classes = []
    self.class_to_idx = {}
    super(VideoSegmentationDataset, self).__init__(
      root, mode, resize_mode, resize_shape, tw, max_temporal_gap
    )

  def read_sample(self, sample):
    sample = super().read_sample(sample)
    sample[INFO]["num_objects"] = sample[TARGETS]["mask"].max()
    return sample

  def read_target(self, sample):
    """
    Reads the segmentation map for the frames given in ``sample['targets']``.
    Returns masks as a `TxHxWxC` array, where `T` is the number of frames,
    `HxW` are the spatial dimensions, and `C` is the number of masks.
    The return value is a dict.

    :param sample: dict describing the sample
    :return: dict with target ``masks``
    """
    masks = map(lambda x: imread(x, pilmode="P"), sample['targets'])
    return {"mask": np.stack([m for m in masks])[..., None]}


class VideoClassificationDataset(VideoDataset):
  def __init__(self, root, num_classes, mode="train", resize_mode=None, resize_shape=None,
               tw=8, max_temporal_gap=8):
    self.num_classes = num_classes
    self.classes = []
    self.class_to_idx = {}

    super(VideoClassificationDataset, self).__init__(
      root, mode, resize_mode, resize_shape, tw, max_temporal_gap
    )

  def read_target(self, sample):
    return {"class": sample[TARGETS]}
