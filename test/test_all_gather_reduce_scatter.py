import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    world_size = xm.xrt_world_size()
    assert world_size >= 128, "this should be tested on a v3-128 or v3-256"
    t1 = torch.randn(199665, device=xm.xla_device())
    t2 = xm.all_gather(t1).flatten()
    t3 = xm.reduce_scatter(xm.REDUCE_SUM, t2, scale=1.0, scatter_dim=0, shard_count=world_size)
    t4 = t3.sum()
    xm.mark_step()
    print(f"t4: {t4}")

if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=(), nprocs=8)
