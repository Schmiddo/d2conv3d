import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_lightning as pl

from omegaconf import DictConfig

from utils.construct import build_network, build_losses, build_optimizer, build_scheduler
from utils.util import partition

from loss.util import register_modulation_saving_hook

from experiments.inference.handler import SaliencyInferenceHandler

from .util import avg_frame_iou


class SaliencyModel(nn.Module):
  def __init__(self, hparams: DictConfig):
    super(SaliencyModel, self).__init__()

    self.hparams = hparams

    self.model = build_network(hparams.model)
    self.loss_fn, self.extra_losses = build_losses(hparams.loss)
    if "saliency" in self.extra_losses:
      self.modulation_maps = register_modulation_saving_hook(self.model)

    self.inference_handler = SaliencyInferenceHandler(self.hparams, self.loss_fn)

  def forward(self, input):
    return self.model(input)

  def configure_optimizers(self):
    if self.hparams.train.decrease_deformable_layer_lr:
      param_groups = [
        {"params": p.values()}
        for p in partition(lambda n, _: "deform_param" in n, self)
      ]
      param_groups[0]["lr"] = 0.1 * self.hparams.solver.lr
      print(f"[SOLVER] Using lr of {param_groups[0]['lr']:.2e} "
            f"for {len(param_groups[0]['params'])} offset layers")
    else:
      param_groups = self.parameters()
    optimizer = build_optimizer(param_groups, self.hparams.solver)
    scheduler = build_scheduler(optimizer, self.hparams.scheduler)
    return [optimizer], [scheduler]

  def _get_prediction(self, logits):
    if "cce" in self.hparams.loss.name:
      pred = logits.argmax(dim=1)
    else:
      pred = logits.sigmoid() > 0.5
    return pred

  def _forward_clip(self, batch):
    clip = batch["images"]
    mask = batch["targets"]["mask"].squeeze(-1)

    if self.hparams.experiment.task == "saliency":
      mask[mask != 0] = 1

    output = self.forward(clip)
    logits = output["logits"] if isinstance(output, dict) else output

    logits = F.interpolate(logits, mask.shape[-3:], mode="trilinear", align_corners=False)
    #mask = F.interpolate(mask.unsqueeze(1), logits.shape[-3:], mode="nearest").squeeze(1)

    if "cce" in self.hparams.loss.name:
      mask = mask.long()
    elif "ce" in self.hparams.loss.name:
      logits = logits.squeeze(1)
      mask = mask.float()

    loss = self.loss_fn(logits, mask)

    if isinstance(loss, dict):
      extra_losses = loss
      loss = extra_losses.pop("total_loss")
    else:
      extra_losses = {}
    if "saliency" in self.extra_losses:
      saliency_loss = self.extra_losses["saliency"](self.modulation_maps, mask)
      extra_losses["saliency"] = saliency_loss.detach()
      # TODO: make loss weight schedule configurable
      saliency_loss_weight = 0.1
      loss += saliency_loss_weight * saliency_loss

    logits = logits.detach()
    pred = self._get_prediction(logits)
    if "iou" in extra_losses:
      iou = extra_losses.pop("iou")
    else:
      iou = avg_frame_iou(pred, mask, num_classes=max(2,self.hparams.model.num_classes), ignore_idx=0)

    return logits, pred, loss, extra_losses, iou

  def training_step(self, batch, batch_idx):
    _, pred, loss, extra_losses, iou = self._forward_clip(batch)
    #if (batch_idx) % 2000 == 0:
    #  self.trainer.log_video(batch["images"], batch["targets"]["mask"], pred, "train")
    return {"loss": loss, "iou": iou, **extra_losses}

  def validation_step(self, batch, batch_idx):
    _, __, loss, extra_losses, iou = self._forward_clip(batch)
    return {"loss": loss, "iou": iou, **extra_losses}

  def test_step(self, batch, batch_idx):
    return self.inference_handler.infer_sequence(self.model, batch)
