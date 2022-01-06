import hydra
from omegaconf import DictConfig

import os
from pytorch_lightning.utilities.seed import seed_everything

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

# Populate hydras config store
import config.defaults

from trainer import Trainer

from utils.construct import build_datasets
from utils.util import load_weights
from utils.distributed import run_distributed


seed_everything(int(os.environ.get("SEED", "12345")))


@hydra.main(config_path="config", config_name="base_config")
def main(cfg: DictConfig):
  print("Building registry... ", end="", flush=True)
  # Imports to populate registry
  import datasets
  import network.backbone
  import network.segmentation
  import network.modules
  import loss
  import experiments
  print("Done.")

  # make sure task + name are set before we do anything else
  print(f"{cfg.experiment.task} - {cfg.name} - {cfg.mode}")

  # Hack -- make nice somehow
  from config.hack import init_from_hydra
  init_from_hydra(cfg)

  #trainer = Trainer(cfg)

  train_data, val_data, test_data = build_datasets(cfg.data)

  if cfg.experiment.task == "classification":
    from experiments import ClassificationModel
    model = ClassificationModel(cfg)
  elif cfg.experiment.task == "saliency" or cfg.experiment.task == "segmentation":
    from experiments import SaliencyModel
    model = SaliencyModel(cfg)
  elif cfg.experiment.task == "panoptic_segmentation":
    from experiments import PanopticModel
    model = PanopticModel(cfg)
  elif cfg.experiment.task == "distillation":
    from experiments import Distillery
    model = Distillery(cfg)
  else:
    raise ValueError(f"Task {cfg.experiment.task} is not implemented.")

  if cfg.model.weights:
    print(f"[MODEL] Loading model weights from {cfg.model.weights}")
    load_weights(model, cfg.model.weights)

  if cfg.mode == "train":
    from dist_fit import fit
    run_distributed(fit, (cfg, model, train_data, val_data, test_data))
    #trainer.fit(model, train_data, val_data,test_data)
  elif cfg.mode == "test":
    trainer = Trainer(cfg)
    trainer.infer(model, test_data)
  else:
    if cfg.mode == "visualize":
      from tools.drop_offsets import visualize_offsets
      visualize_offsets(cfg, model, val_data)
    elif cfg.mode == "export":
      from tools.export import export
      export(cfg, model)
    elif cfg.mode == "inspect":
      from tools.inspect_weights import inspect
      inspect(cfg, model)
    print(f"'mode' must be 'train' or 'test', but is {cfg.mode}")


if __name__ == "__main__":
  main()
