import torch.nn as nn


class Experiment(nn.Module):
  def __init__(self):
    super(Experiment, self).__init__()
    self.current_epoch = 0

  def set_epoch(self, current_epoch):
    self.current_epoch = current_epoch
