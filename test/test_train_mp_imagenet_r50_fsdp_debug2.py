import args_parse


SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

MODEL_OPTS = {
    "--model": {"choices": SUPPORTED_MODELS, "default": "resnet50"},
    "--test_set_batch_size": {"type": int},
    "--lr_scheduler_type": {"type": str},
    "--lr_scheduler_divide_every_n_epochs": {"type": int},
    "--lr_scheduler_divisor": {"type": int},
    "--num_warmup_epochs": {"type": int},
    "--test_only_at_end": {"action": "store_true"},
    "--ckpt_dir": {"type": str, "default": "Please specify"},
    "--ckpt_interval": {"type": int, "default": 5},
    "--eval_interval": {"type": int, "default": 5},
    "--reshard_after_forward": {"action": "store_true"},
    "--flatten_parameters": {"action": "store_true"},
    "--use_all_gather_via_all_reduce": {"action": "store_true"},
    '--use_nested_fsdp': {'action': 'store_true'},
    '--rng_seed': {"type": int, "default": 42},
    # AMP only works with XLA:GPU
    "--amp": {"action": "store_true"},
    # Using zero gradients optimization for AMP
    "--use_zero_grad": {"action": "store_true"},
}

FLAGS = args_parse.parse_common_options(
    datadir="/tmp/imagenet",
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

import torch
import torch.nn as nn
import torchvision
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel as FSDP


def train_imagenet():
    device = xm.xla_device()

    xm.master_print("==> Preparing data..")
    img_dim = 224
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(
            torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
            torch.zeros(FLAGS.batch_size, dtype=torch.int64),
        ),
        sample_count=train_dataset_len // FLAGS.batch_size // xm.xrt_world_size(),
    )
    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    xm.rendezvous("data loading completed")
    xm.master_print("data loading completed")

    torch.manual_seed(FLAGS.rng_seed)
    model = getattr(torchvision.models, FLAGS.model)().to(device)
    # always wrap the base model with an outer FSDP
    model = FSDP(
        model,
        reshard_after_forward=FLAGS.reshard_after_forward,
        flatten_parameters=FLAGS.flatten_parameters,
        use_all_gather_via_all_reduce=FLAGS.use_all_gather_via_all_reduce,
    )
    xm.rendezvous("FSDP model init completed")
    xm.master_print("FSDP model init completed")

    loss_fn = nn.CrossEntropyLoss()

    def train_loop_fn(loader, epoch):
        model.train()
        xm.master_print(f"training epoch {epoch}")
        for step, (data, target) in enumerate(loader):
            xm.master_print(f"running epoch {epoch} step {step}/{len(loader)}")
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

    xm.rendezvous("training begins")
    xm.master_print("training begins")
    train_loop_fn(train_device_loader, 0)


def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type("torch.FloatTensor")
    train_imagenet()


if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
