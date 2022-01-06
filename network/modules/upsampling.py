import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register


@register("module")
class UpsamplerBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.ConvTranspose3d(
      in_channels, out_channels, 2, stride=2, bias=True)
    # self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

  def forward(self, input):
    output = self.conv(input)
    # output = self.bn(output)
    return F.relu(output)
