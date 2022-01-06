import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ._lovasz import lovasz_hinge, lovasz_softmax

from utils.registry import register


@register("loss", "lovasz_ce")
class LovaszHingeLoss(nn.Module):
  def __init__(self):
    super(LovaszHingeLoss, self).__init__()

  def forward(self, logits, labels):
    return checkpoint(lovasz_hinge, logits, labels)


@register("loss", "lovasz_cce")
class LovaszSoftmaxLoss(nn.Module):
  def __init__(self):
    super(LovaszSoftmaxLoss, self).__init__()

  def forward(self, logits, labels):
    return lovasz_softmax(logits.softmax(dim=1), labels)
