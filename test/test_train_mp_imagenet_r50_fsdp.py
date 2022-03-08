import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--num_warmup_epochs': {
        'type': int,
    },
    '--test_only_at_end': {
        'action': 'store_true',
    },
    '--ckpt_dir': {
        'type': str, "default": "Please specify",
    },
    '--ckpt_interval': {
        'type': int, "default": 5,
    },
    '--eval_interval': {
        'type': int, "default": 5,
    },
    '--reshard_after_forward': {
      'action': 'store_true',
    },
    '--flatten_parameters': {
      'action': 'store_true',
    },
    # AMP only works with XLA:GPU
    '--amp': {
        'action': 'store_true',
    },
    # Using zero gradients optimization for AMP
    '--use_zero_grad': {
        'action': 'store_true',
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

import os
import schedulers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from torch_xla.amp import autocast, GradScaler

from fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel as FSDP

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.1,
                'lr_scheduler_divide_every_n_epochs': 30,
                'lr_scheduler_divisor': 10,
                'num_warmup_epochs': 5,
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)


def get_model_property(key):
  default_model_property = {
      'img_dim': 224,
      'model_fn': getattr(torchvision.models, FLAGS.model)
  }
  model_properties = {
      'inception_v3': {
          'img_dim': 299,
          'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
      },
  }
  model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
  return model_fn


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def broadcast_xla_master_model_param(model):
  """
  Broadcast the model parameters from master process to other processes
  """
  parameters_and_buffers = []
  is_master = xm.is_master_ordinal(local=False)
  for p in chain(model.parameters(), model.buffers()):
    # Set all params in non-master devices to zero so that all_reduce is
    # equivalent to broadcasting parameters from master to other devices.
    if not is_master:
      zero = torch.tensor(0, dtype=p.data.dtype, device=p.data.device)
      p.data.mul_(zero)
    parameters_and_buffers.append(p.data)
  xm.wait_device_ops()
  xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
  xm.mark_step()
  xm.rendezvous("broadcast_xla_master_model_param")


def train_imagenet():
  print('==> Preparing data..')
  img_dim = get_model_property('img_dim')
  if FLAGS.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size //
        xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_len = len(train_dataset.imgs)
    resize_dim = max(img_dim, 256)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'val'),
        # Matches Torchvision's eval transforms except Torchvision uses size
        # 256 resize for all models both here and in the train loader. Their
        # version crashes during training on 299x299 images, e.g. inception.
        transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(img_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler, test_sampler = None, None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
      test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=test_sampler,
        drop_last=FLAGS.drop_last,
        shuffle=False,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  device = xm.xla_device()
  model = get_model_property('model_fn')().to(device)
  model = FSDP(
    model,
    reshard_after_forward=FLAGS.reshard_after_forward,
    flatten_parameters=FLAGS.flatten_parameters,
  )
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(FLAGS.logdir)
  optimizer = optim.SGD(
      model.parameters(),
      lr=FLAGS.lr,
      momentum=FLAGS.momentum,
      weight_decay=1e-4)
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * xm.xrt_world_size())
  lr_scheduler = schedulers.WarmupAndExponentialDecayScheduler(
      optimizer,
      num_steps_per_epoch=num_training_steps_per_epoch,
      divide_every_n_epochs=FLAGS.lr_scheduler_divide_every_n_epochs,
      divisor=FLAGS.lr_scheduler_divisor,
      num_warmup_epochs=5,
      summary_writer=writer)
  loss_fn = nn.CrossEntropyLoss()
  if FLAGS.amp:
    scaler = GradScaler(use_zero_grad=FLAGS.use_zero_grad)

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      if FLAGS.amp:
        with autocast():
          output = model(data)
          loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        # do not reduce the gradients (since we are using sharded params)
        scaler.step(optimizer)
        scaler.update()
      else:
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        # do not reduce the gradients (since we are using sharded params)
        optimizer.step()

      tracker.add(FLAGS.batch_size)
      if lr_scheduler:
        lr_scheduler.step()
      if step % FLAGS.log_steps == 0:
        xm.add_step_closure(
            _train_update, args=(device, step, loss, tracker, epoch, writer))

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    for step, (data, target) in enumerate(loader):
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]
      if step % FLAGS.log_steps == 0:
        xm.add_step_closure(
            test_utils.print_test_update, args=(device, None, epoch, step))
    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
  for epoch in range(1, FLAGS.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
    run_eval = (
        (epoch % FLAGS.eval_interval == 0 and not FLAGS.test_only_at_end) or
        epoch == FLAGS.num_epochs)
    if run_eval:
      accuracy = test_loop_fn(test_device_loader, epoch)
      xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
          epoch, test_utils.now(), accuracy))
      max_accuracy = max(accuracy, max_accuracy)
      test_utils.write_to_summary(
          writer,
          epoch,
          dict_to_write={'Accuracy/test': accuracy},
          write_xla_metrics=True)
    if epoch % FLAGS.ckpt_interval == 0 or epoch == FLAGS.num_epochs:
      rank = xm.get_ordinal()
      world_size = xm.xrt_world_size()
      ckpt = {
        "model": model.state_dict(),
        "shard_metadata": model.get_shard_metadata(),
        "optimizer": optimizer.state_dict(),
      }
      ckpt_file = os.path.join(
        FLAGS.ckpt_dir,
        f"checkpoint-{epoch}_rank-{rank:08d}-of-{world_size:08d}.pth"
      )
      xm.save(ckpt, ckpt_file, global_master=True)
      xm.master_print(f"checkpoint saved to {ckpt_file}")
    if FLAGS.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_imagenet()
  if accuracy < FLAGS.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
