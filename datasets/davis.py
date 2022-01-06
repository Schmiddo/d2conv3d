import glob
import os.path as osp
import random as rand

from datasets.base_dataset import VideoSegmentationDataset, INFO, IMAGES_, TARGETS

from utils.registry import register


@register("dataset")
class DAVIS(VideoSegmentationDataset):
  def __init__(self, root, mode="train", resize_mode=None, resize_shape=None,
               clip_size=8, max_tw=16, imset=None):
    if imset and imset not in ("2016", "2017"):
      raise ValueError(f"'imset' must be either 2016 or 2017 but is {imset}")
    if mode not in ("train", "val", "test"):
      raise ValueError(f"'mode' should be either train, val, or test but is {mode}")
    self.imset = imset or "2017"
    self.image_dir = osp.join(root, "JPEGImages", "480p")
    self.mask_dir = osp.join(root, "Annotations_unsupervised", "480p")
    self.num_frames = {}
    super(DAVIS, self).__init__(root, mode, resize_mode, resize_shape, clip_size, max_tw)

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

  def _create_sample(self, video, num_frames, support_indices):
    sample = {}
    sample[IMAGES_] = [osp.join(self.image_dir, video, f"{s:05}.jpg") for s in support_indices]
    sample[TARGETS] = [osp.join(self.mask_dir, video, f"{s:05}.png") for s in support_indices]

    sample[INFO] = {
      "support_indices": support_indices,
      "video": video,
      "num_frames": num_frames,
    }
    return sample

  def create_sample_list(self):
    imset_path = osp.join(self.root, "ImageSets", self.imset, f"{self.mode}.txt")
    self.videos = [l.strip("\n") for l in open(imset_path, "r")]
    for video in self.videos:
      img_list = sorted(glob.glob(osp.join(self.image_dir, video, "*.jpg")))
      num_frames = len(img_list)
      self.num_frames[video] = num_frames

      for i, img in enumerate(img_list):
        support_indices = self._get_support_indices(i, video)
        sample = self._create_sample(video, num_frames, support_indices)

        self.raw_samples.append(sample)
    self.samples = self.raw_samples


@register("dataset")
class DAVISInfer(DAVIS):
  def __init__(self, root, mode="val", resize_mode=None, resize_shape=None,
               clip_size=8, imset=None):
    super(DAVISInfer, self).__init__(root, mode, resize_mode, resize_shape, clip_size, clip_size, imset)

  def create_sample_list(self):
    imset_path = osp.join(self.root, "ImageSets", self.imset, f"{self.mode}.txt")
    self.videos = [l.strip("\n") for l in open(imset_path, "r")]
    for video in self.videos:
      img_list = sorted(glob.glob(osp.join(self.image_dir, video, "*.jpg")))
      num_frames = len(img_list)
      self.num_frames[video] = num_frames

      support_indices = list(range(num_frames))
      sample = self._create_sample(video, num_frames, support_indices)

      self.raw_samples.append(sample)
    self.samples = self.raw_samples
