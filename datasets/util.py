import random

import numpy as np
from imgaug import augmenters as iaa

TRANSLATION = 0.1
SHEAR = 0.05
ROTATION = 15
FLIP = 40


def generate_clip_from_image(raw_frame, raw_mask, temporal_window, **kwargs):
  global TRANSLATION, ROTATION, SHEAR
  if 'translation' in kwargs:
    TRANSLATION = kwargs['translation']
  if 'rotation' in kwargs:
    ROTATION = kwargs['rotation']
  if 'shear' in kwargs:
    SHEAR = kwargs['shear']

  clip_frames = np.repeat(raw_frame[np.newaxis], temporal_window, axis=0)
  clip_masks = np.repeat(raw_mask[np.newaxis], temporal_window, axis=0)
  # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
  # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
  # image.
  sometimes = lambda aug: iaa.Sometimes(0.05, aug)
  blur = sometimes(iaa.OneOf([
    iaa.GaussianBlur((0.0, 0.5)),
    # iaa.AverageBlur(k=(2, 7)),
    # iaa.MedianBlur(k=(3, 11)),
  ]))
  seq = iaa.Sequential([
    # iaa.Fliplr(FLIP / 100.),  # horizontal flips
    sometimes(iaa.ElasticTransformation(alpha=(200, 220), sigma=(17.0, 19.0))),
    iaa.Affine(
      scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
      translate_percent={"x": (-TRANSLATION, TRANSLATION), "y": (-TRANSLATION, TRANSLATION)},
      rotate=(-ROTATION, ROTATION),
      shear=(-SHEAR, SHEAR),
      mode='edge',
    )
  ], random_order=True)

  frame_aug = raw_frame[np.newaxis]
  mask_aug = raw_mask[np.newaxis]
  # create sequence of transformations of the current image
  for t in range(temporal_window - 1):
    frame_aug, mask_aug = seq(images=frame_aug.astype(np.uint8), segmentation_maps=mask_aug.astype(np.uint8))
    frame_aug = blur(images=frame_aug)
    clip_frames[t + 1] = frame_aug[0]
    clip_masks[t + 1] = mask_aug[0]

  return clip_frames, clip_masks


def flip_video_sample(sample, p_horizontal=0.5, p_vertical=0.0, p_temporal=0.0):
  # D H W C
  assert sample["images"].ndim == 4

  if random.random() < p_horizontal:
    sample["images"] = sample["images"][:, :, ::-1, :]
    sample["targets"]["mask"] = sample["targets"]["mask"][:, :, ::-1, :]
  if random.random() < p_vertical:
    sample["images"] = sample["images"][:, ::-1, :, :]
    sample["targets"]["mask"] = sample["targets"]["mask"][:, ::-1, :, :]
  if random.random() < p_temporal:
    sample["images"] = sample["images"][::-1, :, :, :]
    sample["targets"]["mask"] = sample["targets"]["mask"][::-1, :, :, :]

  return sample


def crop_video_sample(sample, ratio=(0.8, 1.0)):
  # D H W C
  assert sample["images"].ndim == 4

  _, width, height, __ = sample["images"].shape
  assert 0 < ratio[0] <= ratio[1] <= 1
  ratio = ratio[0] + random.random() * (ratio[1] - ratio[0])
  crop_width, crop_height = int(ratio * width), int(ratio * height)

  left, bottom = int(random.random() * (width - crop_width)), int(random.random() * (height - crop_height))
  right, top = left + crop_width, bottom + crop_height

  sample["images"] = sample["images"][:, left:right, bottom:top, :]
  sample["targets"]["mask"]  = sample["targets"]["mask"][:, left:right, bottom:top, :]

  return sample
