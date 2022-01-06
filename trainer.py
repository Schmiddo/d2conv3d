import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

import einops

import os
import os.path as osp
import time
import signal

from utils.average_meter import AverageMeterDict
from utils.scoreboard import Scoreboard
from utils.util import find_most_recent_checkpoint, fill_segmentation_map, color_map
from utils.distributed import (
  batch_to_device, detach, reduce_dict,
  init_distributed, cleanup_distributed,
)

_cmap = color_map()


# TODO: find more elegant solution for workers to not save checkpoints?
# Workaround so dataloader workers do not save checkpoints on signals.
def _worker_init_fn(_):
  def _handler(_, __):
    pass
  signal.signal(signal.SIGINT, _handler)
  signal.signal(signal.SIGTERM, _handler)
  signal.signal(signal.SIGUSR1, _handler)


class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg
    self.max_epochs = cfg.train.num_epochs
    self.max_steps = cfg.train.num_steps
    self.eval_every_n_epochs = cfg.train.eval_every_n_epochs

    self.start_epoch = 0
    self.current_epoch = 0
    self.global_step = 0

    self.save_every_n_steps = cfg.train.save_every_n_steps
    self.train_scoreboard = Scoreboard(["loss", "acc", "iou"])
    self.eval_scoreboard = Scoreboard(["acc", "iou"])
    self.hooks = cfg.train.hooks

    self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    self.world_size = int(os.environ.get("WORLD_SIZE", 1))

    self._delete_checkpoint = None
    self.register_signal_handler()

    torch.backends.cudnn.benchmark = cfg.train.cudnn_benchmark

    self.scaler = None

    self.log_path = osp.join(cfg.paths.log_dir, cfg.experiment.task, cfg.name)
    self.writer = None

  def _init_dataloaders(self, train_data, val_data, test_data):
    if self.world_size > 1:
      self.train_sampler = DistributedSampler(train_data, shuffle=True)
      self.val_sampler = DistributedSampler(val_data, shuffle=False)
      self.test_sampler = DistributedSampler(test_data, shuffle=False)
    else:
      self.train_sampler = None
      self.val_sampler = None
      self.test_sampler = None

    self.train_loader = DataLoader(
      train_data,
      shuffle=(self.train_sampler is None),
      sampler=self.train_sampler,
      batch_size=self.cfg.data.batch_size,
      num_workers=self.cfg.data.num_workers,
      pin_memory=self.cfg.data.pin_memory,
      worker_init_fn=_worker_init_fn
    )
    self.val_loader = DataLoader(
      val_data,
      shuffle=False,
      sampler=self.val_sampler,
      batch_size=self.cfg.data.batch_size,
      num_workers=self.cfg.data.num_workers,
      pin_memory=self.cfg.data.pin_memory,
      worker_init_fn=_worker_init_fn
    )
    self.test_loader = DataLoader(
      test_data, shuffle=False, sampler=self.test_sampler,
      batch_size=1, pin_memory=self.cfg.data.pin_memory
    )

  def _init_model(self):
    # TODO: add to config
    if "oob_tracking" in self.hooks and self.local_rank == 0:
      from tools.hooks.oob_tracking import register_oob_tracking_hooks
      register_oob_tracking_hooks(self, self.model)
    # TODO: feels incredibly hacky and unsafe, is there a better way?
    self.training_step = self.model.training_step
    self.validation_step = self.model.validation_step
    self.test_step = self.model.test_step
    if torch.cuda.is_available():
      self.model.cuda()
      if self.world_size > 1:
        if self.cfg.train.use_sync_batchnorm:
          self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.local_rank])

  def fit(self, model, train_data, val_data, test_data):
    if self.world_size > 1:
      init_distributed()

    if self.local_rank == 0 and self.cfg.logging.tensorboard:
      log_dir = osp.join(self.cfg.paths.log_dir, self.cfg.experiment.task, self.cfg.name)
      self.writer = SummaryWriter(log_dir)
      print(f"Tensorboard logs in {log_dir}")
    if self.cfg.train.precision == 16:
      self.scaler = GradScaler()

    if self.local_rank == 0:
      print(f"Training with {self.cfg.train.precision} bit precision")

    self.model = model
    self.configure_optimizers = model.configure_optimizers
    self.model.trainer = self

    self._init_model()
    self._init_dataloaders(train_data, val_data, test_data)

    self.optimizer, self.lr_scheduler = self.configure_optimizers()
    self.optimizer = self.optimizer[0]
    self.lr_scheduler = self.lr_scheduler[0]

    ckpt = find_most_recent_checkpoint(self.cfg.paths.checkpoint_dir)
    if ckpt:
      if self.local_rank == 0:
        print(f"Resuming training from {ckpt}")
      self.load_checkpoint(ckpt)
      del ckpt
    else:
      if self.local_rank == 0:
        print(f"Starting to train for {self.cfg.train.num_epochs} epochs.")

    for epoch in range(self.start_epoch, self.max_epochs):
      self.current_epoch = epoch
      results = self._train(self.model, self.train_loader)

      if self.local_rank == 0:
        improvements = self.train_scoreboard.update(results)
        if improvements and self.cfg.train.save_best_training_model:
          self.save_checkpoint("model_best_train.pth")

      # Step LR schedulers after epoch
      self.lr_scheduler.step()

      if self.cfg.train.save_every_n_epochs and ((epoch + 1) % self.cfg.train.save_every_n_epochs) == 0:
        self.save_checkpoint(f"epoch_{epoch + 1}.pth")

      if ((epoch + 1) % self.eval_every_n_epochs) == 0:
        results = self._eval(self.model, self.val_loader)

        if self.local_rank == 0:
          self.log_results(results, "val/")
          improvements = self.eval_scoreboard.update(results)
          if improvements and self.cfg.train.save_best_eval_model:
            self.save_checkpoint("model_best_eval.pth")

    self.save_checkpoint(f"{self.global_step + 1}.ckpt")
    if self.cfg.train.run_inference_after_fit:
      results = self._test(self.model, self.test_loader)

      if self.local_rank == 0:
        self.log_results(results, "test/")
    cleanup_distributed()

  # Run single training epoch
  def _train(self, model, data):
    model.train()
    result_averages = AverageMeterDict()

    if self.train_sampler:
      self.train_sampler.set_epoch(self.current_epoch)
    # TODO: workaround due to lightning
    if hasattr(self.model, "module"):
      self.model.module.current_epoch_ = self.current_epoch
    else:
      self.model.current_epoch_ = self.current_epoch

    start = time.time()
    for idx, batch in enumerate(data):
      batch = batch_to_device(batch, "cuda")

      self.optimizer.zero_grad()
      with autocast(self.cfg.train.precision != 32):
        results = self.training_step(batch, idx)

      loss = results["loss"]
      if torch.isnan(loss):
        if self.local_rank == 0:
          print("Loss is nan, aborting")
        cleanup_distributed()
        exit(0)
      if self.scaler:
        self.scaler.scale(loss).backward()

        if self.cfg.train.grad_clip_type != "none":
          self.scaler.unscale_(self.optimizer)
          if self.cfg.train.grad_clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_max)
          elif self.cfg.train.grad_clip_type == "value":
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.train.grad_clip_max)

        self.scaler.step(self.optimizer)
        self.scaler.update()
      else:
        loss.backward()
        self.optimizer.step()

      if self.cfg.logging.gradients and self.local_rank == 0 and self.global_step % 100 == 0:
        for n, p in self.model.named_parameters():
          if p.grad is not None:
            self.writer.add_histogram(f"grads/{n}", p.grad, global_step=self.global_step)
      if self.cfg.logging.weights and self.local_rank == 0 and self.global_step % 100 == 0:
        for n, p in self.model.named_parameters():
          if p.grad is not None:
            self.writer.add_histogram(f"weights/{n}", p, global_step=self.global_step)

      results = detach(results)
      if self.world_size > 1:
        reduce_dict(results, 0)
        for t in results.values():
          t.div_(self.world_size)
      torch.cuda.synchronize()
      results["Time"] = time.time() - start
      start = time.time()
      result_averages.update(results)
      if self.local_rank == 0:
        self.log_results(results, "train/")
        print(f"[Iter: {self.global_step}][{idx}/{len(data)}] {result_averages}")

        if (self.global_step + 1) % self.save_every_n_steps == 0:
          self.save_checkpoint(f"{self.global_step + 1}.ckpt")

      self.global_step += 1
    return result_averages.avg

  def _eval_test_loop(self, model, data, step_func, tag):
    model.eval()
    result_averages = AverageMeterDict()

    start = time.time()
    for idx, batch in enumerate(data):
      batch = batch_to_device(batch, "cuda")
      with autocast(self.cfg.train.precision != 32):
        with torch.no_grad():
          results = step_func(batch, idx)

      results = detach(results)
      if self.world_size > 1:
        reduce_dict(results, 0)
        for t in results.values():
          t.div_(self.world_size)

      results["Time"] = time.time() - start
      start = time.time()
      result_averages.update(results)

      if self.local_rank == 0:
        print(f"{tag} [{idx}/{len(data)}] {result_averages}")
    return result_averages.avg

  # Run evaluation between epochs
  def _eval(self, model, data):
    return self._eval_test_loop(model, data, self.validation_step, "Validation")

  # Run test/inference
  def _test(self, model, data):
    return self._eval_test_loop(model, data, self.test_step, "Test")

  def infer(self, model, data):
    self.model = model
    self._init_model()
    data = DataLoader(data, shuffle=False, batch_size=1, pin_memory=True)
    results = self._test(model, data)
    if self.local_rank == 0:
      print(f"Inference finished.")
      for k, v in results.items():
        v = v.item() if isinstance(v, torch.Tensor) else v
        print(f"{k:35}: {v:7.5f}")

  def log_results(self, results, prefix=""):
    if self.local_rank != 0 or self.writer is None:
      return
    # TODO: allow nested tags/values in results
    for tag, val in results.items():
      if tag.lower() == "time":
        continue
      self.writer.add_scalar(prefix + tag, val, self.global_step)

  def log_video(self, video, mask, pred, namespace):
    global _cmap
    if self.local_rank != 0 or self.writer is None:
      return
    _cmap = _cmap.to(mask.device)
    mask = fill_segmentation_map(mask.squeeze(-1), _cmap)
    pred = fill_segmentation_map(pred.squeeze(-1), _cmap)
    video = einops.rearrange(video, "b c d h w -> b d c h w")
    mask = einops.rearrange(mask, "b d h w c -> b d c h w")
    pred = einops.rearrange(pred, "b d h w c -> b d c h w")
    self.writer.add_video(f"{namespace}/samples/video", video, fps=2, global_step=self.global_step)
    self.writer.add_video(f"{namespace}/samples/mask", mask, fps=2, global_step=self.global_step)
    self.writer.add_video(f"{namespace}/samples/pred", pred, fps=2, global_step=self.global_step)

  def register_signal_handler(self):
    def _handler(sig, _):
      if sig == signal.SIGUSR1:
        signal.signal(sig, signal.SIG_IGN)
      print(f"Received signal {sig}, shutting down. I'm rank {self.local_rank}.")
      if self.local_rank == 0 and self.global_step > 100:
        self.save_checkpoint(f"{self.global_step}.ckpt", remove_after_load=True)
      cleanup_distributed()
      exit(0)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGUSR1, _handler)

  def save_checkpoint(self, save_name, remove_after_load=False):
    if self.local_rank == 0:
      save_name = osp.join(self.cfg.paths.checkpoint_dir, save_name)
      if not osp.exists(self.cfg.paths.checkpoint_dir):
        os.makedirs(self.cfg.paths.checkpoint_dir)
      print(f"Saving checkpoint to {save_name}")
      ckpt = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        "scaler": self.scaler.state_dict() if self.scaler else None,
        "epoch": self.current_epoch,
        "global_step": self.global_step,
        "train_scoreboard": self.train_scoreboard.state_dict(),
        "eval_scoreboard": self.eval_scoreboard.state_dict(),
        "remove": remove_after_load
      }
      torch.save(ckpt, save_name)

      if self._delete_checkpoint:
        try:
          os.remove(self._delete_checkpoint)
        except:
          print(f"Failed to remove checkpoint {self._delete_checkpoint}")
        else:
          print(f"Removed old checkpoint {self._delete_checkpoint}")
        finally:
          self._delete_checkpoint = None

  def load_checkpoint(self, checkpoint_name):
    ckpt = torch.load(checkpoint_name, map_location="cpu")
    if not hasattr(self.model, "module"):
      model_state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    else:
      model_state_dict = {k if k.startswith("module.") else "module." + k: v for k, v in ckpt["model"].items()}
    self.model.load_state_dict(model_state_dict)
    self.optimizer.load_state_dict(ckpt["optimizer"])
    if self.lr_scheduler and ckpt["scheduler"]:
      self.lr_scheduler.load_state_dict(ckpt["scheduler"])
    if self.scaler and ckpt["scaler"]:
      self.scaler.load_state_dict(ckpt["scaler"])
    self.start_epoch = self.current_epoch = ckpt["epoch"]
    self.global_step = ckpt["global_step"] + 1

    if "train_scoreboard" in ckpt:
      self.train_scoreboard.load_state_dict(ckpt["train_scoreboard"])
      self.eval_scoreboard.load_state_dict(ckpt["eval_scoreboard"])

    if ckpt.get("remove", False):
      self._delete_checkpoint = checkpoint_name
