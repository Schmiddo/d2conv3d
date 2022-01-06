import math
import operator as op

from utils.util import Font


class Scoreboard:
  _metrics = {
    "loss": "lower",
    "acc": "higher",
    "iou": "higher",
  }

  def __init__(self, save_on):
    save_on = {k for k in save_on if k in self._metrics}
    self.is_better = {
      k: op.lt if self._metrics[k] == "lower" else op.gt
      for k in save_on
    }
    self.current_best = {
      k: math.inf if self._metrics[k] == "lower" else -math.inf
      for k in save_on
    }

  @staticmethod
  def _pretty_print(improvements):
    s = []
    for k, v in improvements.items():
      s.append(f"{Font.bold(k)}: {v['old']:.5f} -> {v['new']:.5f}")
    if s:
      print(" | ".join(s))
    else:
      print("No improvements to last epoch.")

  def update(self, new_values):
    improvements = {}
    for k in (self.current_best.keys() & new_values.keys() & self.is_better.keys()):
      b = self.current_best[k]
      v = new_values[k]
      if self.is_better[k](v, b):
        improvements[k] = {"new": v, "old": b}
        self.current_best[k] = v
    self._pretty_print(improvements)
    return improvements

  def state_dict(self):
    return {"current_best": self.current_best}

  def load_state_dict(self, state_dict):
    self.current_best.update(state_dict["current_best"])
