import random
from enum import Enum, unique
import numpy as np
from imgaug import imresize_many_images, imresize_single_image


"""
All resize methods expect a dict of tensors containing at least an entry ``images``.
Supported formats are NxHxWxC (multiple images/single video) and HxWxC (single image).
All input tensors are expected to have the same spatial dimensions.
"""


@unique
class ResizeMode(Enum):
  FIXED_SIZE = "fixed_size"
  UNCHANGED = "unchanged"
  RANDOM_RESIZE_AND_CROP = "random_resize_and_crop"
  RANDOM_RESIZE_AND_OBJECT_CROP = "random_resize_and_object_crop"
  RESIZE_AND_OBJECT_CROP = "resize_and_object_crop"
  BBOX_CROP_AND_RESIZE_FIXED_SIZE = "bbox_crop_and_resize_fixed_size"
  RESIZE_SHORT_EDGE = "resize_short_edge"
  RESIZE_SHORT_EDGE_AND_CROP = "resize_short_edge_and_crop"
  RANDOM_RESIZE_FIVE_CROP = "random_resize_five_crop"
  CENTER_CROP = "center_crop"

  @classmethod
  def _missing_(cls, value):
    return cls.UNCHANGED


def _tuple(t):
  if type(t) == int:
    return (t,) * 2
  if len(t) == 1:
    return (t[0],) * 2
  assert len(t) == 2
  return tuple(t)


def _get_spatial_size(t):
  return t.shape[:2] if t.ndim <= 3 else t.shape[1:3]


def _resize_single_tensor(t, size, mode=None):
  return imresize_single_image(t, size, mode) if t.ndim <= 3 else imresize_many_images(t, size, mode)


def _crop_single_tensor(t, left, bot, width, height):
  if t.ndim <= 3:
    return t[bot:bot+height, left:left+width]
  else:
    return t[:, bot:bot+height, left:left+width]


def resize(tensors, resize_mode, size):
  if isinstance(resize_mode, str):
    resize_mode = ResizeMode(resize_mode)
  if resize_mode is None or resize_mode == ResizeMode.UNCHANGED:
    if tensors['images'].max() <= 1:
      # rescale the mage tensor values to be within 0-255. This would make it consistent with other resize modes
      tensors['images'] = tensors['images'] * 255.0
    return tensors
  crop_size = _tuple(size) if size is not None else None
  if resize_mode == ResizeMode.RANDOM_RESIZE_AND_CROP:
    return random_resize_and_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RANDOM_RESIZE_AND_OBJECT_CROP:
    return random_resize_and_object_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_AND_OBJECT_CROP:
    return resize_and_object_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.FIXED_SIZE:
    return resize_fixed_size(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_SHORT_EDGE:
    return resize_short_edge_to_fixed_size(tensors, crop_size)
  elif resize_mode == ResizeMode.RESIZE_SHORT_EDGE_AND_CROP:
    return resize_short_edge_and_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RANDOM_RESIZE_FIVE_CROP:
    return random_resize_five_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.CENTER_CROP:
    return center_crop(tensors, crop_size)
  else:
    assert False, ("resize mode not implemented yet", resize_mode)


def random_resize_five_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=size)
  tensors_resized = random_five_crop(tensors_resized, size)
  return tensors_resized


def random_five_crop(tensors, crop_size):
  assert "images" in tensors
  h, w = _get_spatial_size(tensors["images"])
  new_h, new_w = crop_size

  region = random.randrange(5)
  if region == 0:
    left, bot = 0, 0
  elif region == 1:
    left, bot = 0, h - new_h
  elif region == 2:
    left, bot = w - new_w, h - new_h
  elif region == 3:
    left, bot = w - new_w, 0
  else:
    left, bot = (w - new_w) // 2, (h - new_h) // 2

  tensors_cropped = {}
  for key, tensor in tensors.items():
    tensors_cropped[key] = _crop_single_tensor(tensor, left, bot, new_w, new_h)

  return tensors_cropped


def center_crop(tensors, crop_size):
  """
  Crop a patch of size ``crop_size`` from the center of input tensors.

  :param tensors: input dict
  :param crop_size: crop size
  :return: dict with cropped tensors
  """
  assert "images" in tensors
  h, w = _get_spatial_size(tensors["images"])
  new_h, new_w = crop_size

  bot = (h - new_h) // 2
  left = (w - new_w) // 2

  tensors_cropped = {}
  for key, tensor in tensors.items():
    tensors_cropped[key] = _crop_single_tensor(tensor, left, bot, new_w, new_h)

  return tensors_cropped


def random_resize_and_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=size)
  tensors_resized = random_crop_tensors(tensors_resized, size)
  return tensors_resized


def random_resize_and_object_crop(tensors, size):
  tensors_resized = resize_random_scale_with_min_size(tensors, min_size=size)
  tensors_resized = random_object_crop_tensors(tensors_resized, size)
  return tensors_resized


def resize_and_object_crop(tensors, size):
  tensors_resized = resize_fixed_size(tensors, size)
  tensors_resized = random_object_crop_tensors(tensors_resized, size)
  return tensors_resized


def resize_short_edge_and_crop(tensors, size):
  tensors_resized = resize_short_edge_to_fixed_size(tensors, size)
  tensors_resized = random_object_crop_tensors(tensors_resized, (size[0], size[1]))
  return tensors_resized


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  """
  Scales tensors choosen randomly from the given range of scales, but ensures that each spatial
  dimension is at least ``min_size``.

  :param tensors: dict of tensors
  :param min_size: minimum spatial dimensions (height, width)
  :param min_scale: lower bound for scale factor
  :param max_scale: upper bound for scale factor (except when image would still not reach minimum size)
  :return: dict with scaled tensors
  """
  img = tensors["images"]

  h, w = _get_spatial_size(img)
  min_h, min_w = min_size
  min_scale_factor = max(min_h/h, min_w/w, 1.0)

  min_scale = np.max([min_scale, min_scale_factor])
  max_scale = np.max([max_scale, min_scale_factor])
  scale_factor = random.uniform(min_scale, max_scale)

  scaled_size = (int(h * scale_factor), int(w * scale_factor))

  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def scale_with_min_size(tensors, min_size):
  """
  Scales tensors so that each spatial dimension is at least ``min_size``.

  :param tensors: dict of tensors
  :param min_size: minimum spatial dimension (height, width)
  :return: dict with scaled tensors
  """
  img = tensors["images"]

  h, w = _get_spatial_size(img)
  min_h, min_w = min_size
  # only resize if one dimension is smaller than the minimum
  scale_factor = max(min_h/h, min_w/w, 1.0)

  scaled_size = (int(h * scale_factor), int(w * scale_factor))

  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_short_edge_to_fixed_size(tensors, size):
  """
  Scales tensors in dict so that the shorter side matches the smaller value of `size`.

  :param tensors: dict of tensors
  :param size: target size
  :return: dict of resized tensors
  """
  img = tensors["images"]
  h, w = _get_spatial_size(img)

  shorter_side = np.min([h, w])
  scale_factor = np.min(size) / shorter_side
  scaled_size = np.around(np.array((h, w)) * scale_factor).astype(np.int32)

  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def random_crop_tensors(tensors, crop_size):
  """
  Randomly crop a region of size ``crop_size`` from input tensors.

  :param tensors: input dict
  :param crop_size: crop size
  :return: dict with cropped tensors
  """
  assert "images" in tensors
  h, w = _get_spatial_size(tensors["images"])
  new_h, new_w = crop_size

  bot = int(random.uniform(0, max(h - new_h, 0)))
  left = int(random.uniform(0, max(w - new_w, 0)))

  tensors_cropped = {}
  for key, tensor in tensors.items():
    tensors_cropped[key] = _crop_single_tensor(tensor, left, bot, new_w, new_h)

  return tensors_cropped


def random_object_crop_tensors(tensors, crop_size):
  """
  Randomly crop a region of size `crop_size` from  images + masks in `tensors`,
  such that objects fully fit into the cropped region.
  If the object is larger than `crop_size`, the region is resized to `crop_size`.

  Expects keys ``images`` and ``mask``.
  Tensors must be in [Dx]HxW[xC] ordering.

  :param tensors: dict w/ keys `images` and `mask`
  :param crop_size: (height, width) of cropped region
  :return: dict with cropped and resized tensors
  """
  assert "images" in tensors and 'mask' in tensors

  img = tensors["images"]
  h, w = _get_spatial_size(img)
  new_h, new_w = crop_size

  mask = tensors["mask"]
  if mask.sum() > 0:
    obj_indices = np.where(mask != 0)
    x, y = (1, 0) if mask.ndim == 2 else (2, 1)
    obj_h_max = np.max(obj_indices[y])
    obj_h_min = np.min(obj_indices[y])
    obj_w_max = np.max(obj_indices[x])
    obj_w_min = np.max(obj_indices[x])
  else:
    obj_h_min = obj_h_max = obj_w_min = obj_w_max = 0

  top_lower_bound = max(0, obj_h_max - new_h)
  left_lower_bound = max(0, obj_w_max - new_w)
  top_upper_bound = min(max(0, (h - new_h)), obj_h_min)
  left_upper_bound = min(max(0, w - new_w), obj_w_min)

  bot = int(random.uniform(top_lower_bound, top_upper_bound))
  left = int(random.uniform(left_lower_bound, left_upper_bound))

  tensors_cropped = {}
  for key, tensor in tensors.items():
    tensors_cropped[key] = _crop_single_tensor(tensor, left, bot, new_w, new_h)
    if tensors_cropped[key].shape != crop_size:
      mode = "nearest" if key.endswith("mask") else "linear"
      tensors_cropped[key] = _resize_single_tensor(tensors_cropped[key], crop_size, mode)
  return tensors_cropped


def resize_fixed_size(tensors, size):
  """
  Scales all tensors in dict to a fixed spatial size.
  Interpolation mode is 'nearest' if the key ends with 'mask', otherwise 'linear'.

  :param tensors: dict of tensors
  :param size: target size
  :return: dict of resized tensors
  """
  tensors_resized = {}
  for key in tensors.keys():
    mode = "nearest" if key.endswith("mask") else "linear"
    tensors_resized[key] = _resize_single_tensor(tensors[key], size, mode)
  return tensors_resized
