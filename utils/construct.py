import torch.nn
from torch.utils.data import DataLoader

from omegaconf import OmegaConf as oc
from inspect import signature, isclass

from utils.registry import retrieve, register
from utils.util import load_weights, Font
from utils.replace import (
  add_weight_standardisation,
  convert_batchnorm,
  add_deformable_conv
)


def _gather_params(cls, cfg, ignore=None):
  params = {}
  missing_params = []
  func_params = signature(cls.__init__ if isclass(cls) else cls).parameters
  for param in func_params:
    if param == "self": continue
    if ignore and param in ignore:
      continue
    if param in cfg:
      params[param] = cfg[param]
    else:
      missing_params.append(f"{param} ({func_params[param].default})")

  if missing_params:
    print(
      f"Building {Font.red(cls.__name__)} using defaults for"
      f" {Font.red(', '.join(missing_params))}"
    )

  return params


def build_cls(cls, **options):
  kwargs = _gather_params(cls, options)
  return cls(**kwargs)


def build_network(cfg):
  return retrieve("network", cfg.name)(cfg)


def get_module_fn(name):
  return retrieve("module", name)


def build_backbone(cfg, **kwargs):
  backbone_fn = retrieve("backbone", cfg.name)
  print(f"[BACKBONE] Constructing backbone as {Font.bold(backbone_fn.__name__)}")
  backbone_options = {
    "strides": oc.to_container(cfg.strides),
    "dilations": oc.to_container(cfg.dilations)
  }
  backbone = backbone_fn(**backbone_options)

  if cfg.weights:
    print(f"[BACKBONE] Loading backbone weights from {Font.bold(cfg.weights)}")
    load_weights(
      backbone,
      cfg.weights,
      ignore_keys=["fc.weight", "fc.bias"],
      strict=False
    )

  if cfg.norm_layer_eps:
    print(f"[BACKBONE] Setting batchnorm epsilon to {cfg.norm_layer_eps:2e}")
    for n, m in backbone.named_modules():
      if isinstance(m, torch.nn.BatchNorm3d):
        m.eps = cfg.norm_layer_eps

  if cfg.replace_norm_layer:
    if cfg.norm_layer in ["GroupNorm", "BCNorm", "EstimatedBCNorm"]:
      extra_args = {"num_groups": cfg.norm_layer_groups}
    else:
      extra_args = {}
    backbone, n_conversions = convert_batchnorm(cfg.norm_layer, backbone, **extra_args)
    print(f"[BACKBONE] Replaced {Font.bold(n_conversions)} norm layers with {Font.bold(cfg.norm_layer)}")

  if cfg.use_ws:
    backbone, n_conversions = add_weight_standardisation(backbone)
    print(f"[BACKBONE] Added WS to {Font.bold(n_conversions)} layers")

  if cfg.freeze_layers:
    print(f"[BACKBONE] Freezing weights for layers {Font.bold(', '.join(cfg.freeze_layers))}")
    for key in cfg.freeze_layers:
      backbone.__getattr__(key).requires_grad_(False)

  if cfg.deformable:
    for key in cfg.deformable.layers:
      layer = backbone.__getattr__(key)
      layer, n_conversions = add_deformable_conv(layer, **cfg.deformable)
      backbone.__setattr__(key, layer)
      print(
        f"[BACKBONE] Replaced {Font.bold(n_conversions)} conv layers in '{Font.bold(key)}'"
        f" with {Font.bold(cfg.deformable.version)} layers"
      )

  return backbone


def build_losses(cfg):
  cls = retrieve("loss", cfg.name)
  loss = cls(**_gather_params(cls, cfg))
  extra_losses = {}
  for el in cfg.extra:
    if "name" in el:
      extra_name = el.name
      extra_kwargs = el
    else:
      extra_name = el
      extra_kwargs = {}
    cls = retrieve("loss", extra_name)
    extra_losses[extra_name] = cls(**_gather_params(cls, extra_kwargs))
  return loss, extra_losses


def build_optimizer(params, cfg):
  cls = retrieve("solver", cfg.name)
  return cls(params, **_gather_params(cls, cfg, ignore=["params"]))


def build_scheduler(optimizer, cfg):
  cls = retrieve("scheduler", cfg.name)
  return cls(optimizer, **_gather_params(cls, cfg, ignore=["optimizer"]))


def build_datasets(cfg):
  train_set_cls = retrieve("dataset", cfg.train.name)
  eval_set_cls = retrieve("dataset", cfg.eval.name)
  test_set_cls = retrieve("dataset", cfg.test.name)

  train_set = build_cls(train_set_cls, cache=cfg.cache, mode="train", **cfg.train)
  eval_set = build_cls(eval_set_cls, cache=cfg.cache, mode="val", **cfg.eval)
  # TODO: replace 'val' with 'test'?
  test_set = build_cls(test_set_cls, cache=cfg.cache, mode="val", **cfg.test)
  return train_set, eval_set, test_set
