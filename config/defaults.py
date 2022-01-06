from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple

from hydra.conf import ConfigStore
from omegaconf import MISSING


@dataclass
class PathConfig:
  data_dir: str = "${env:DATA,./data}"
  work_dir: str = "${env:WORK,./}"
  save_dir: str = "${paths.work_dir}/saved_models"
  checkpoint_dir: str = "${paths.work_dir}/runs/${experiment.task}_${name}/checkpoint"
  log_dir: str = "${paths.work_dir}/log"


@dataclass
class DatasetConfig:
  name: str = MISSING
  root: str = MISSING
  annotation_path: Optional[str] = None
  imset: Optional[str] = None
  clip_size: int = MISSING
  frames_between_clips: Optional[int] = None
  max_tw: Optional[int] = None
  resize_mode: Optional[str] = None
  resize_shape: Optional[List] = None
  restricted_image_category_list: Optional[List] = field(default_factory=lambda: [])


@dataclass
class ConcatDatasetConfig(DatasetConfig):
  dataset_names: List = MISSING
  options: List = MISSING
  weights: Optional[List] = None
  num_samples: Optional[int] = None


_coco_restricted_image_category_list = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
  'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'snowboard',
  'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
  'surfboard', 'tennis racket', 'remote', 'cell phone'
]

YVOS_COCO_DAVISConfigNode = ConcatDatasetConfig(
  name="Concat",
  root="",
  clip_size=8,
  max_tw=12,
  resize_mode="resize_short_edge_and_crop",
  resize_shape=[480,854],
  dataset_names=["DAVIS", "COCO", "YoutubeVOS"],
  options=[
    {"root": "${paths.data_dir}/DAVIS-Unsupervised/DAVIS"},
    {"root": "${paths.data_dir}/coco", "restricted_image_category_list": _coco_restricted_image_category_list},
    {"root": "${paths.data_dir}/youtube-vos"}
  ],
  weights=[0.3, 0.25, 0.45]
)


DAVISConfigNode = DatasetConfig(
  name="DAVIS",
  imset="2017",
  root="${paths.data_dir}/DAVIS-Unsupervised/DAVIS",
  resize_mode="fixed_size",
  resize_shape=[480, 854],
  clip_size=8,
  max_tw=8,
)

COCOConfigNode = DatasetConfig(
  name="COCO",
  root="${paths.data_dir}/coco",
  resize_mode="resize_short_edge_and_crop",
  resize_shape=[480, 854],
  clip_size=8,
  max_tw=8,
  restricted_image_category_list=_coco_restricted_image_category_list,
  imset="2014"
)

YoutubeVOSConfigNode = DatasetConfig(
  name="YoutubeVOS",
  root="${paths.data_dir}/youtube-vos",
  resize_mode="resize_short_edge_and_crop",
  resize_shape=[480, 854],
  clip_size=8,
  max_tw=8,
)


@dataclass
class DataConfig:
  cache: str = "${paths.work_dir}/cache"
  batch_size: int = 32
  num_workers: int = 4
  pin_memory: bool = True
  # TODO: implement augmentations
  augmentations: List = field(default_factory=lambda: [])
  train: DatasetConfig = MISSING
  eval: DatasetConfig = MISSING
  test: DatasetConfig = MISSING


@dataclass
class OptimizerConfig:
  name: str = MISSING


@dataclass
class SGDConfig(OptimizerConfig):
  name: str = "sgd"
  lr: float = 1e-3
  momentum: float = 0.9
  weight_decay: float = 1e-5


@dataclass
class AdamConfig(OptimizerConfig):
  name: str = "adam"
  lr: float = 1e-5
  amsgrad: bool = False


@dataclass
class AdamWConfig(OptimizerConfig):
  name: str = "adamw"
  lr: float = 1e-5
  weight_decay: float = 1e-2
  amsgrad: bool = False


@dataclass
class SchedulerConfig:
  name: str = "step"
  step_size: int = 10
  gamma: float = 0.1


@dataclass
class LossConfig:
  name: str = "cce"
  fraction: float = 0.25
  warmup: Optional[str] = None
  warmup_steps: int = 2000
  extra: List = field(default_factory=lambda: [])


@dataclass
class DeformableConvConfig:
  version: str = "DCNv1"
  layers: List = field(default_factory=lambda: [])
  offset_groups: int = 1
  replace_1x1_kernels: bool = False
  activation: str = "sigmoid"
  init: str = "zero"
  conditioning: str = "none"
  size_activation: str = "elu"
  norm_type: str = "none"
  norm_groups: int = 1


_default_dilations = [1, 1, 1, 1]
_default_strides =   [1, 2, 2, 2]
# _reduced_ts_[dilations,strides] are dilations & strides
# of the ResNetNoTS implementation
_reduced_ts_dilations = [1, 1, [2, 1, 1], [2, 1, 1]]
_reduced_ts_strides =   [1, 2, [1, 2, 2], [1, 2, 2]]
_no_ts_dilations = [1, [2, 1, 1], [2, 1, 1], [2, 1, 1]]
_no_ts_strides =   [1, [1, 2, 2], [1, 2, 2], [1, 2, 2]]
@dataclass
class BackboneConfig:
  name: str = MISSING
  dilations: List = field(default_factory=lambda: _default_dilations)
  strides: List = field(default_factory=lambda: _default_strides)
  freeze_layers: List = field(default_factory=lambda: [])
  deformable: Optional[DeformableConvConfig] = None
  replace_norm_layer: bool = True
  norm_layer: str = "FrozenBatchNorm"
  norm_layer_eps: Optional[float] = None
  norm_layer_groups: Optional[int] = 32
  use_ws: bool = False
  weights: str = "${paths.save_dir}/backbones/r3d18.pth"


@dataclass
class ModelConfig:
  name: str = MISSING
  weights: str = ""
  backbone: BackboneConfig = BackboneConfig()
  use_ws: bool = False
  num_classes: int = MISSING
  tw: int = 16
  mean: (float, float, float) = (0.43216, 0.394666, 0.37645)
  std: (float, float, float) = (0.22803, 0.22145, 0.216989)


@dataclass
class DecoderConfig:
  name: str = "SaliencyDecoder"
  num_classes: int = "${model.num_classes}"
  mdim: int = "${model.mdim}"
  inter_block: str = "${model.inter_block}"
  connect_block: str = "Identity"
  refine_block: str = "${model.refine_block}"

  conv_type: str = "Conv3d"
  norm_type: str = "GroupNorm"
  norm_groups: int = 32
  activation: str = "relu"
  extra_backbone_norm: bool = False
  interpolation_mode: str = "trilinear"
  align_corners: bool = False

  deformable: Optional[DeformableConvConfig] = None
  fpn: bool = True


@dataclass
class SaliencyModelConfig(ModelConfig):
  name: str = "EncoderDecoderNetwork"
  num_classes: int = 1
  tw: int = 8

  mdim: int = 256

  inter_block: str = "C3D"
  refine_block: str = "Refine3dSimple"

  decoder: DecoderConfig = DecoderConfig()


@dataclass
class TrainConfig:
  num_epochs: Optional[int] = 10
  num_steps: Optional[int] = None
  eval_every_n_epochs: int = 1
  save_every_n_steps: int = 500000
  save_every_n_epochs: Optional[int] = None
  save_best_training_model: bool = False
  save_best_eval_model: bool = True
  # only relevant if model uses BN somewhere
  use_sync_batchnorm: bool = True
  run_inference_after_fit: bool = False
  precision: int = 32
  cudnn_benchmark: bool = True
  hooks: List = field(default_factory=lambda: [])
  decrease_deformable_layer_lr: bool = False

  grad_clip_type: str = "none"
  grad_clip_max: Optional[float] = None

  checkpoint_backbone: bool = True


@dataclass
class InferenceConfig:
  temporal_gap: int = 5
  num_clips: Optional[int] = None
  save_path: str = "results"
  save_logits: bool = False
  save_predictions: bool = True


@dataclass
class LoggingConfig:
  tensorboard: bool = True
  cmdline: bool = True
  gradients: bool = False
  weights: bool = False


@dataclass
class ExperimentConfig:
  task: str = MISSING


@dataclass
class Config:
  paths: PathConfig = PathConfig()
  logging: LoggingConfig = LoggingConfig()

  experiment: ExperimentConfig = ExperimentConfig()
  mode: str = "train"
  name: str = MISSING

  train: TrainConfig = TrainConfig()
  infer: InferenceConfig = InferenceConfig()

  data: DataConfig = DataConfig()
  model: ModelConfig = ModelConfig()
  loss: LossConfig = LossConfig()
  solver: OptimizerConfig = OptimizerConfig()
  scheduler: SchedulerConfig = SchedulerConfig()

  defaults: List = field(default_factory=lambda: _defaults)


_defaults = [
  {"paths": "default"},
  {"logging": "default"},

  {"experiment": "???"},

  {"train": "default"},
  {"infer": "default"},

  {"data": "${defaults.2.experiment}_default"},
  {"data/dataset@data/train": "${defaults.2.experiment}_default"},
  {"data/dataset@data/eval": "${defaults.2.experiment}_default"},
  {"data/dataset@data/test": "${defaults.2.experiment}_default"},

  {"model": "${defaults.2.experiment}_default"},
  {"model/backbone": "r3d18"},
  {"model/backbone/deformable": "non_deformable"},
  {"model/decoder/deformable": "${defaults.2.experiment}_default", "optional": True},

  {"loss": "${defaults.2.experiment}_default"},
  {"solver": "${defaults.2.experiment}_default"},
  {"scheduler": "default"},

  {"hydra/run": "rundir"},

  {"": "glue"},
  {"": "${defaults.2.experiment}_glue", "optional": True}
]

cs = ConfigStore.instance()

cs.store(group="schema/model/backbone", name="resnet",
         node=BackboneConfig, package="model/backbone")
cs.store(group="schema/model", name="SaliencyModel",
         node=SaliencyModelConfig, package="model")

cs.store(group="paths", name="env", node=PathConfig)
# 'default' for 'paths' should be created as default.yaml in config/paths
cs.store(group="logging", name="default", node=LoggingConfig)

cs.store(group="experiment", name="saliency",
         node=ExperimentConfig(task="saliency"))

cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="infer", name="default", node=InferenceConfig)

cs.store(group="data", name="saliency_default", node=DataConfig(batch_size=2))

cs.store(group="data/dataset", name="DAVIS", node=DAVISConfigNode)

cs.store(group="data/dataset", name="COCO", node=COCOConfigNode)
cs.store(group="data/dataset", name="YoutubeVOS", node=YoutubeVOSConfigNode)

cs.store(group="data/dataset", name="yvos_coco_davis", node=YVOS_COCO_DAVISConfigNode)

# TODO: solve this with defaults list in Hydra 1.1
cs.store(group="data/dataset", name="saliency_default", node=DAVISConfigNode)

cs.store(group="model", name="saliency_default", node=SaliencyModelConfig)

cs.store(group="model/backbone/deformable", name="non_deformable", node=None)

cs.store(group="model/backbone/ts", name="default", package="model/backbone",
         node={"strides": _default_strides, "dilations": _default_dilations})
cs.store(group="model/backbone/ts", name="reduced_ts", package="model/backbone",
         node={"strides": _reduced_ts_strides, "dilations": _reduced_ts_dilations})
cs.store(group="model/backbone/ts", name="no_ts", package="model/backbone",
         node={"strides": _no_ts_strides, "dilations": _no_ts_dilations})

cs.store(group="model/decoder", name="SaliencyDecoder", node=DecoderConfig)

cs.store(group="model/decoder/deformable", name="saliency_default", node=None)
cs.store(group="model/decoder/deformable", name="non_deformable", node=None)

cs.store(group="loss", name="saliency_default", node=LossConfig(name="ce"))

cs.store(group="solver", name="SGD", node=SGDConfig)
cs.store(group="solver", name="Adam", node=AdamConfig)
cs.store(group="solver", name="AdamW", node=AdamWConfig)
cs.store(group="solver", name="saliency_default", node=AdamConfig)

cs.store(group="scheduler", name="default", node=SchedulerConfig)

# TODO: fix rundir for multirun/slurm jobs
cs.store(group="hydra/run", name="rundir",
         node={"dir": "${paths.work_dir}/runs/${experiment.task}_${name}"})

cs.store(group="hydra/job_logging/handlers/file", name="train_log",
         node={"filename": "${mode}.log"})
cs.store(group="hydra/job_logging/handlers/file", name="test_log",
         node={"filename": "${mode}_${data.train.name}.log"})

_glue_config = """
hydra:
 job_logging:
  handlers:
   file:
    filename: ${mode}.log
data:
 test:
  name: ${data.eval.name}Infer
"""
_saliency_glue_config = """
data:
 test:
  resize_mode: unchanged
  resize_shape: null
"""

cs.store(name="glue", node=_glue_config)
cs.store(name="saliency_glue", node=_saliency_glue_config)

cs.store("base_config", Config)


# Legacy stuff
DAVISConfig = DatasetConfig

