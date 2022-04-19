import torch.nn.functional as F
import torch_xla.core.xla_model as xm


def all_gather_via_all_reduce(value, dim=0, groups=None, pin_layout=True):
  """
  This is the old all_gather implementation via all_reduce in PyTorch XLA 1.10 in
  https://github.com/pytorch/xla/blob/v1.10.0/torch_xla/core/xla_model.py#L583-L615,
  which avoids the GRPC error (see https://github.com/pytorch/xla/issues/3423).
  """
  if dim < 0:
    dim = value.dim() + dim
  size = value.size(dim)
  padding = [0] * (2 * value.dim())
  ordinal = xm.get_ordinal()
  if groups is None:
    left, right = ordinal, xm.xrt_world_size() - 1 - ordinal
  else:
    ordinals = dict()
    for g in groups:
      for i, x in enumerate(g):
        ordinals[x] = (i, len(g) - 1 - i)
    left, right = ordinals[ordinal]
  idx = value.dim() - 1 - dim
  padding[2 * idx] = left * size
  padding[2 * idx + 1] = right * size
  # use in-place all_reduce on padded_value
  padded_value = F.pad(value, padding)
  xm.all_reduce(
      xm.REDUCE_SUM, [padded_value], groups=groups, pin_layout=pin_layout)
  return padded_value
