import torch


def register_canary(model: torch.nn.Module, user_check=None):
  def _canary(name):
    def _canary_hook(module, input, output):
      for n, p in module.named_parameters():
        if torch.any(torch.isnan(p)):
          print(f"nan in parameter {n} of module {name}")
        if torch.any(torch.isinf(p)):
          print(f"inf in parameter {n} of module {name}")
        if user_check and user_check(p):
          print(f"user_check returns true on parameter {n} of module {name}")
      for i, t in enumerate(input):
        if torch.any(torch.isnan(t)):
          print(f"nan in input {i} of module {name}")
        if torch.any(torch.isinf(t)):
          print(f"inf in input {i} of module {name}")
        if user_check and user_check(t):
          print(f"user_check returns true on input {i} of module {name}")
    return _canary_hook

  for name, module in model.named_modules():
    module.register_forward_hook(_canary(name))


def avg_frame_iou(pred, gt, num_classes=None, ignore_idx=None):
  if num_classes is None:
    pred_classes = pred.max() + 1
    mask_classes = gt.max() + 1
    num_classes = max(pred_classes, mask_classes)

  @torch.jit.script
  def class_iou(pred: torch.Tensor, gt: torch.Tensor, c: int):
    pred_c = pred == c
    gt_c = gt == c
    i = (pred_c & gt_c).sum(dim=[2, 3])
    u = (pred_c | gt_c).sum(dim=[2, 3])

    u[u == 0] = 1
    i[u == 0] = 1

    return i.float() / u.float()

  ious = []
  for c in range(num_classes):
    if c == ignore_idx:
      continue
    iou = class_iou(pred, gt, c)
    ious.append(iou)

  return torch.stack(ious).mean()


def video_iou(pred, gt, classes, void_mask=None):
  assert pred.dim() == 4
  assert gt.dim() == 4

  if isinstance(classes, int):
    classes = list(range(classes))

  if void_mask is None:
    void_mask = torch.zeros_like(gt)

  @torch.jit.script
  def class_video_iou(pred:torch.Tensor, gt: torch.Tensor, c: int, void_mask: torch.Tensor):
    pred_c = ((pred == c) & (void_mask == 0))
    gt_c = ((gt == c) & (void_mask == 0))
    i = (pred_c & gt_c).sum(dim=[1, 2, 3])
    u = (pred_c | gt_c).sum(dim=[1, 2, 3])

    u[u == 0] = 1
    i[u == 0] = 1

    return i.float() / u.float()

  ious = []
  for c in classes:
    ious.append(class_video_iou(pred, gt, c, void_mask))
  return torch.stack(ious).mean()
