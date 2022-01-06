import os.path as osp
import numpy as np

from datasets.base_dataset import VideoSegmentationDataset, INFO, IMAGES_, TARGETS

from utils.registry import register
from .util import generate_clip_from_image


@register("dataset")
class COCO(VideoSegmentationDataset):
  def __init__(self, root, mode="train", resize_mode=None, resize_shape=None,
               clip_size=8, max_tw=16, restricted_image_category_list=None, imset="2014"):
    if mode not in ("train", "val"):
      raise ValueError(f"'mode' should be either train, or val but is {mode}")
    if imset not in ("2014", "2017"):
      raise ValueError(f"imset should be either 2014 or 2017 but is {imset}")
    self.imset = imset
    self.image_dir = osp.join(root, mode + self.imset)
    self.annotation_file = osp.join(root, "annotations", f"instances_{mode}{self.imset}.json")
    self.coco = None
    self.annotations = None
    self.anns = []
    self.filename_to_anns = {}
    self.num_frames = {}
    if mode == "train":
      self.filter_crowd_images = True
      self.min_box_size = 30
    else:
      self.filter_crowd_images = False
      self.min_box_size = -1.0
    self.restricted_image_category_list = restricted_image_category_list
    super(COCO, self).__init__(root, mode, resize_mode, resize_shape, clip_size, max_tw)

  def _build_filename_to_anns_dict(self):
    for ann in self.anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']
      if file_name in self.filename_to_anns:
        self.filename_to_anns[file_name].append(ann)
      else:
        self.filename_to_anns[file_name] = [ann]

  def _filter_anns(self):
    # exclude all images which contain a crowd
    if self.filter_crowd_images:
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items()
                               if not any([an["iscrowd"] for an in anns])}
    # filter annotations with too small boxes
    if self.min_box_size != -1.0:
      self.filename_to_anns = {f: [ann for ann in anns if ann["bbox"][2] >= self.min_box_size and ann["bbox"][3]
                                   >= self.min_box_size] for f, anns in self.filename_to_anns.items()}

    # remove annotations with crowd regions
    self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                             for f, anns in self.filename_to_anns.items()}
    # restrict images to contain considered categories
    if self.restricted_image_category_list:
      print("filtering images to contain categories", self.restricted_image_category_list)
      restricted_cat_ids = set(
        c["id"] for c in self.coco.cats.values()
        if c["name"] in self.restricted_image_category_list
      )
      self.filename_to_anns = {
        f: [ann for ann in anns if ann["category_id"] in restricted_cat_ids]
          for f, anns in self.filename_to_anns.items()
      }
      # filter out images without annotations
      self.filename_to_anns = {f: anns for f, anns in self.filename_to_anns.items() if len(anns) > 0}

      for cat_id in restricted_cat_ids:
        n_imgs_for_cat = sum([1 for anns in self.filename_to_anns.values()
          if any([ann["category_id"] == cat_id for ann in anns])])
        print(f"number of images containing {self.label_map[cat_id - 1]['name']}: {n_imgs_for_cat}")

    n_before = len(self.anns)
    self.anns = []
    for anns in self.filename_to_anns.values():
      self.anns += anns
    n_after = len(self.anns)
    print("filtered annotations:", n_before, "->", n_after)

  def create_sample_list(self):
    import pycocotools.coco as coco
    self.coco = coco.COCO(self.annotation_file)
    ann_ids = self.coco.getAnnIds()
    self.anns = self.coco.loadAnns(ann_ids)
    self.label_map = {k-1: v for k, v in self.coco.cats.items()}

    self._build_filename_to_anns_dict()
    self._filter_anns()

    imgs = [
      osp.join(self.image_dir, fn)
      for fn in self.filename_to_anns.keys()
    ]
    for line in imgs:
      video = line.split("/")[-1].split(".")[0]
      self.videos.append(video)
      self.num_frames[video] = 1
      sample = {
        IMAGES_: [line],
        TARGETS: [],
        INFO: {
          "support_indices": self.tw * [1],
          "video": video,
          "num_frames": 1,
        }
      }
      self.raw_samples.append(sample)
    self.samples = self.raw_samples

  def read_target(self, sample):
    img_filename = sample[IMAGES_][0]
    anns = self.filename_to_anns[img_filename.split("/")[-1]]
    img = self.coco.loadImgs(anns[0]['image_id'])[0]

    height = img['height']
    width = img['width']
    #sample[INFO]['shape'] = (height, width)

    label = np.zeros((height, width, 1))
    for i, ann in enumerate(anns):
      mask = self.coco.annToMask(ann)[:, :, None]
      label[mask != 0] = i + 1
    num_objects = len(np.unique(label))
    if num_objects == 1:
      print("GT contains only background.")

    return [label.astype(np.uint8)]

  def read_sample(self, sample):
    image = self.read_image(sample)[0]
    mask = self.read_target(sample)[0]
    image, mask = generate_clip_from_image(image, mask, self.tw)
    sample[IMAGES_] = image
    sample[TARGETS] = {"mask": mask}
    sample[INFO]["shape"] = self._get_spatial_dim(sample[IMAGES_])
    sample[INFO]["num_objects"] = len(np.unique(mask))
    return sample
