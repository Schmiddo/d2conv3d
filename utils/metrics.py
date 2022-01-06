import torch


class Accuracy:
  def __init__(self):
    self.tps = 0
    self.num = 0

  @torch.no_grad()
  def __call__(self, gt, pred):
    tps = (gt == pred).sum()
    num = gt.numel()
    self.tps += tps
    self.num += num
    return  tps / num

  def total(self):
    return self.tps / self.num


class MAE:
  def __init__(self):
    self.error_sum = 0
    self.num = 0

  @torch.no_grad()
  def __call__(self, gt, pred):
    if not gt.shape == pred.shape:
      raise ValueError(f"Expected gt and pred to have same shape, but got"
                       f"{gt.shape} and {pred.shape}")
    error_sum = (gt - pred).abs().sum()
    num = gt.numel()
    self.error_sum += error_sum
    self.num += num
    return error_sum / num

  def total(self):
    return self.error_sum / self.num


class F1:
  def __init__(self):
    self.tp = 0
    self.fnp = 0

  @torch.no_grad()
  def __call__(self, gt, pred):
    if not gt.shape == pred.shape:
      raise ValueError(f"Expected gt and pred to have same shape, but got"
                       f"{gt.shape} and {pred.shape}")
    gt, pred = gt.bool(), pred.bool()
    tp = (gt & pred).sum()
    fnp = (gt ^ pred).sum()
    self.tp += tp
    self.fnp += fnp
    if tp == 0 and fnp == 0:
      return 0
    return 2 * tp / (2 * tp + fnp)

  def total(self):
    if self.tp == 0 and self.fnp == 0:
      return 0
    return 2 * self.tp / (2 * self.tp + self.fnp)


if __name__ == "__main__":
  from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

  a = torch.randint(0, 2, (3, 4, 5))
  b = torch.rand((3, 4, 5))
  c = 2 * b**2

  def test_metric(expected_fn, fn, gts, preds):
    expected_vals = [expected_fn(gt.flatten().numpy(), pred.flatten().numpy())
                     for gt, pred in zip(gts, preds)]
    vals = [fn(gt, pred) for gt, pred in zip(gts, preds)]
    print("Max diff", torch.as_tensor([(ev - v) for ev, v in zip(expected_vals, vals)]).max().item())

    total_gt = torch.cat([gt.flatten() for gt in gts]).numpy()
    total_pred = torch.cat([pred.flatten() for pred in preds]).numpy()
    expected_total = expected_fn(total_gt, total_pred)
    total = fn.total()

    W = 8
    print(f"{'expected':>{W}} {'result':>{W}} {'diff':>{W}}")
    print(f"{expected_total:{W}.5f} {total.item():{W}.5f} {(expected_total - total).item():{W}.5f}")

  test_metric(mean_absolute_error, MAE(), [a, a], [b, c])

  b, c = (b > 0.5), (c > 0.5)
  test_metric(f1_score, F1(), [a, a], [b, c])

  test_metric(accuracy_score, Accuracy(), [a, a], [b, c])
