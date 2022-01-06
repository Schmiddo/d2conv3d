from torch.utils.data import Dataset, Subset

import math
import random as rand

from utils.registry import retrieve, register
from utils.construct import build_cls


class ConcatDataset(Dataset):
  def __init__(self, datasets, weights=None, num_samples=None):
    self.datasets = datasets
    self.lengths = [len(d) for d in datasets]
    self.weights = weights or [1.0 / len(datasets)] * len(datasets)
    if abs(1 - sum(self.weights)) > 1e-6:
      raise ValueError("Weights should sum up to 1.0")

    self.id_mapping = []
    self.samples_per_dataset = []

    num_samples = num_samples or sum(self.lengths)
    print(f"{num_samples} available")

    for i, (wt, ds) in enumerate(zip(self.weights, self.datasets)):
      assert 0. < wt <= 1.
      num_samples_ds = int(round(wt * num_samples))
      if num_samples_ds < len(ds):
        ds = Subset(ds, rand.sample(range(len(ds)), num_samples_ds))

      repetitions = int(math.floor(num_samples_ds / len(ds)))
      idxes = list(range(len(ds))) * repetitions
      idxes += rand.sample(range(len(ds)), num_samples_ds - len(idxes))

      self.id_mapping.extend([(i, j) for j in idxes])
      self.samples_per_dataset.append(num_samples_ds)

    self.num_samples = sum(self.samples_per_dataset)
    print(f"Using {self.num_samples} samples")

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    ds, idx = self.id_mapping[idx]
    return self.datasets[ds][idx]


@register("dataset", "Concat")
def build_concat_dataset(dataset_names, options, weights, num_samples=None,
      clip_size=8, frames_between_clips=16, max_tw=16,
      resize_mode=None, resize_shape=None):
  assert len(dataset_names) == len(options)
  assert len(dataset_names) == len(weights)
  datasets = []
  for i, dname in enumerate(dataset_names):
    dataset_class = retrieve("dataset", dname)
    datasets.append(build_cls(dataset_class, clip_size=clip_size,
        frames_between_clips=frames_between_clips, max_tw=max_tw,
        resize_mode=resize_mode, resize_shape=resize_shape, **options[i]))
  return ConcatDataset(datasets, weights, num_samples)
