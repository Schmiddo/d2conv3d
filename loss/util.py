import torch
import torch.nn as nn

from dconv_native.deform_conv import DeformConvNd


def register_modulation_saving_hook(model):
  out = {}
  def _make_hook(name):
    def _hook(m, i, o):
      c = o.shape[1] // 4
      offsets, alpha = torch.split(o, [3*c, c], dim=1)
      out[name] = offsets, alpha
    return _hook
  for n, m in model.named_modules():
    if "deform_params" in n:
      m.register_forward_hook(_make_hook(n))
  return out


class MultiTermLoss(nn.Module):
  def __init__(self, losses, weights):
    super(MultiTermLoss, self).__init__()
    assert len(losses) == len(weights)
    self.losses = nn.ModuleDict(losses)
    self.weights = weights

  def forward(self, **kwargs):
    total_loss = kwargs["logits"].new_zeros(1)
    losses = {}
    for l in self.losses:
      loss = self.losses[l](get_loss_args(l, **kwargs))
      total_loss += self.weights[l] * loss
      losses[l] = loss.detach()
    return total_loss, losses


def get_loss_args(loss, **kwargs):
  if loss == "ce":
    a=nn.BCEWithLogitsLoss()
    return {"input": kwargs["logits"], "target": kwargs[""]}