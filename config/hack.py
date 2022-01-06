from dataclasses import dataclass


@dataclass
class GlobalConfig:
  checkpoint_backbone: bool = False


__config = GlobalConfig()


def init_from_hydra(cfg):
  __config.checkpoint_backbone = cfg.train.checkpoint_backbone


def get_global_config():
  return __config
