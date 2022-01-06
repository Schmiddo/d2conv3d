import glob
import os.path as osp
import random as rand
import numpy as np
from imageio import imread

from datasets.base_dataset import VideoSegmentationDataset, INFO, IMAGES_, TARGETS

from utils.registry import register


@register("dataset")
class FBMS(VideoSegmentationDataset):
  def __init__(self, root, mode="train", resize_mode=None, resize_shape=None,
               clip_size=8, max_tw=16):
    if mode not in ("train", "val", "test"):
      raise ValueError(f"'mode' should be either train, val, or test but is {mode}")
    self.image_dir = osp.join(root, "Trainingset" if mode == "train" else "Testset")
    self.mask_dir = osp.join(root, "inst", "train" if mode == "train" else "test")
    self.num_frames = {}
    super(FBMS, self).__init__(root, mode, resize_mode, resize_shape, clip_size, max_tw)

  def read_target(self, sample):
    masks = []
    shape = sample["info"]["shape"]
    for t in sample["targets"]:
      if osp.exists(t):
        mask = imread(t, pilmode="P")
      else:
        mask = np.zeros(shape).astype(np.uint8)
      masks.append(mask)
    return {"mask": np.stack(masks)[..., None]}

  def _get_support_indices(self, index, video):
    temporal_window = self.max_temporal_gap if self.is_train() else self.tw
    start_index = max(0, index - temporal_window//2)
    stop_index = min(self.num_frames[video], index + temporal_window//2)

    indices = list(range(start_index, stop_index))

    if self.is_train():
      # TODO: sample without replacement?
      indices = sorted(rand.choices(indices, k=self.tw))
    else:
      missing_frames = self.tw - len(indices)
      if min(indices) == 0:
        indices = indices + missing_frames * [start_index]
      else:
        indices = missing_frames * [stop_index-1] + indices
    return indices

  def _create_sample(self, video, img_list, mask_list, support_indices):
    sample = {
      IMAGES_: [img_list[s] for s in support_indices],
      TARGETS: [mask_list[s] for s in support_indices],
      INFO: {
        "support_indices": support_indices,
        "video": video,
        "num_frames": self.num_frames[video],
        "gt_frames": np.array([osp.exists(mask_list[s]) for s in support_indices]),
      }
    }
    return sample

  def create_sample_list(self):
    self.videos = [osp.basename(v) for v in glob.glob(self.image_dir + "/*")]
    if len(self.videos) == 0:
      raise ValueError(f"Image directory {self.image_dir} is empty")
    for video in self.videos:
      img_list = sorted(glob.glob(osp.join(self.image_dir, video, "*.jpg")))
      mask_list = [
        osp.join(self.mask_dir, video, (osp.basename(img)[:-4] + ".png"))
        for img in img_list
      ]
      num_frames = len(img_list)
      self.num_frames[video] = num_frames

      for i, img in enumerate(img_list):
        support_indices = self._get_support_indices(i, video)
        sample = self._create_sample(video, img_list, mask_list, support_indices)

        self.raw_samples.append(sample)
    self.samples = self.raw_samples


@register("dataset")
class FBMSInfer(FBMS):
  def __init__(self, root, mode="val", resize_mode=None, resize_shape=None,
               clip_size=8):
    super(FBMSInfer, self).__init__(root, mode, resize_mode, resize_shape, clip_size, clip_size)

  def create_sample_list(self):
    self.videos = [osp.basename(v) for v in glob.glob(self.image_dir + "/*")]
    if len(self.videos) == 0:
      raise ValueError(f"Image directory {self.image_dir} is empty")
    for video in self.videos:
      img_list = sorted(glob.glob(osp.join(self.image_dir, video, "*.jpg")))
      mask_list = [
        osp.join(self.mask_dir, video, (osp.basename(img)[:-4] + ".png"))
        for img in img_list
      ]
      num_frames = len(img_list)
      self.num_frames[video] = num_frames

      support_indices = list(range(num_frames))
      sample = self._create_sample(video, img_list, mask_list, support_indices)

      self.raw_samples.append(sample)
    self.samples = self.raw_samples
