import torch
import numpy as np
import einops
from utils.metrics import MAE, F1

from os import makedirs
import os.path as osp

from utils.video import get_overlapping_clips
from utils.util import save_video, Timer

from experiments.util import avg_frame_iou


class BaseInferenceHandler:
  def __init__(self, cfg):
    self.clip_size = cfg.model.tw
    self.temporal_gap = min(cfg.model.tw, cfg.infer.temporal_gap)

    self.save_path = f"{cfg.infer.save_path}_{self.temporal_gap}"
    self.save_predictions = cfg.infer.save_predictions
    self.save_logits = cfg.infer.save_logits

  @torch.no_grad()
  def infer_sequence(self, model, sequence):
    raise NotImplementedError()


class SaliencyInferenceHandler(BaseInferenceHandler):
  def __init__(self, cfg, loss_fn):
    super(SaliencyInferenceHandler, self).__init__(cfg)
    self.save_path = osp.join(self.save_path, "vos")
    self.loss_fn = loss_fn
    self.mae = MAE()
    self.f1 = F1()

  @torch.no_grad()
  def infer_sequence(self, model, sequence):
    video = sequence["images"].squeeze()
    mask = sequence["targets"]["mask"].squeeze()
    mask[mask != 0] = 1

    num_frames = video.shape[1]

    logits = {i: [] for i in range(num_frames)}
    clips = get_overlapping_clips(
      video, self.clip_size, self.temporal_gap, stretch_last_frame=True
    )

    timer = Timer(synchronize=True, output=False)
    video_name = sequence['info']['video'][0]
    for i, clip in enumerate(clips):
      print(f"{video_name} [{i + 1}/{len(clips)}]")
      with timer:
        clip_logits = model(clip.unsqueeze(0).clone()).squeeze(0)
      clip_logits = einops.rearrange(clip_logits, "c d h w -> d c h w")
      for j in range(self.clip_size):
        f = min(num_frames - 1, i * self.temporal_gap + j)
        logits[f] += [clip_logits[j]]

    if self.save_logits:
      makedirs(self.save_path, exist_ok=True)
      torch.save(logits, f"{self.save_path}/{video_name}_logits.pt")

    # tube stitching
    logits = [torch.stack(logits[i], dim=0).mean(dim=0) for i in range(num_frames)]
    logits = torch.stack(logits)

    (uh, lh), (uw, lw) = sequence['info']['pad']
    H, W = logits.shape[-2:]
    logits = logits[..., lh:H - uh, lw:W - uw]
    mask = mask[:, lh:H - uh, lw:W - uw]

    if logits.shape[1] == 1:
      logits = logits.squeeze(1)
      pred = (logits.sigmoid() > 0.5)
    else:
      pred = logits.argmax(dim=1)

    if self.save_predictions:
      save_video(f"{self.save_path}/{video_name}", pred.cpu(), format="png")

    if "gt_frames" in sequence["info"]:
      # select only frames that have gt available
      frame_mask = sequence["info"]["gt_frames"].squeeze()
      mask = mask[frame_mask]
      logits = logits[frame_mask]
      pred = pred[frame_mask]
    loss = self.loss_fn(logits, mask.float())
    iou = avg_frame_iou(pred.unsqueeze(1), mask.unsqueeze(1))
    mae = self.mae(mask, logits.sigmoid())
    total_mae = self.mae.total()
    f1 = self.f1(mask, pred)
    total_f1 = self.f1.total()
    fps = num_frames / timer.total()

    return {"iou": iou, "loss": loss, "fps": fps, "mae": mae, "total_mae": total_mae, "f1": f1, "total_f1": total_f1}

