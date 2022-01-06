import functools

import torch
from torch import distributed as dist
from torch.distributed import ReduceOp

import torch.multiprocessing as mp

import os


def get_free_port():
  import socket
  from contextlib import closing
  with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


def __run_dist(rank, func, *args):
  os.environ["RANK"] = str(rank)
  os.environ["LOCAL_RANK"] = str(rank)
  func(*args)


def run_distributed(worker_func, args):
  node_size = max(torch.cuda.device_count(), 1)
  local_rank = os.environ.get("RANK")
  if node_size > 1 and local_rank is None:
    print(f"RANK not set, spawning {node_size} processes")
    os.environ["WORLD_SIZE"] = str(node_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    mp.spawn(__run_dist, (worker_func,) + args, node_size)
  else:
    worker_func(*args)


def init_distributed():
  dist.init_process_group("nccl", init_method="env://")
  world_size = int(os.environ["WORLD_SIZE"])
  global_rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])

  torch.cuda.set_device(local_rank)
  print(f"Distributed training, process {global_rank + 1}/{world_size}")


def cleanup_distributed():
  if dist.is_initialized():
    dist.destroy_process_group()


def batch_to_device(batch, device):
  """
  Transfers a tensor, list of tensors, or dict of tensors to a given device.
  Works best with pinned memory. Non-tensors are returned as they are.

  :param batch: list, dict, or single tensor
  :param device: Target device
  :return:
  """
  if isinstance(batch, torch.Tensor):
    return batch.to(device, non_blocking=True)
  elif isinstance(batch, list):
    return [batch_to_device(b, device) for b in batch]
  elif isinstance(batch, dict):
    return {k: batch_to_device(v, device) for k, v in batch.items()}
  else:
    return batch


def detach(obj):
  if isinstance(obj, torch.Tensor):
    return obj.detach()
  elif isinstance(obj, list):
    return [detach(o) for o in obj]
  elif isinstance(obj, dict):
    return {k: detach(v) for k, v in obj.items()}
  else:
    return obj


def reduce_dict(tensor_dict, dst, op=ReduceOp.SUM, group=dist.group.WORLD):
  """
  In-place, synchronous reduction of a dictionary of tensors to one process
  :param tensor_dict: dict of tensors to reduce (in-place)
  :param dst: Destination rank
  :param op: Operation used for element-wise reductions
      (see ``torch.distributed.ReduceOp``)
  :param group: Process group to work on
  :return: None
  """
  _handles = set()
  for t in tensor_dict.values():
    _handles.add(dist.reduce(t, dst, op, group, True))
  for h in _handles:
    h.wait()


def barrier():
  if dist.is_initialized():
    dist.barrier()
