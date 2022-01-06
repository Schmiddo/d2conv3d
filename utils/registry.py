import torch.nn as nn
import torch.optim as optim


_registry = {
  "dataset": {},
  "backbone": {},
  "network": {},
  "module": {},
  "loss": {
    "ce": nn.BCEWithLogitsLoss,
    "cce": nn.CrossEntropyLoss,
  },
  "solver": {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW
  },
  "scheduler": {
    "step": optim.lr_scheduler.StepLR,
    "exp": optim.lr_scheduler.ExponentialLR,
  }
}


def register(collection, key=None):
  def decorator(cls):
    nonlocal key
    if key is None:
      key = cls.__name__
    _registry[collection][key] = cls
    return cls
  return decorator


def print_registry(collection):
  def _print_collection(c):
    print(f"Models registered as '{c}':")
    for k in sorted(_registry[c].keys()):
      print(" ", k)
  if collection in _registry:
    _print_collection(collection)
  else:
    for c in _registry:
      _print_collection(c)


def retrieve(collection, key):
  try:
    return _registry[collection][key]
  except KeyError:
    print_registry(collection)
    raise ValueError(f"{key} not registered as {collection}")
