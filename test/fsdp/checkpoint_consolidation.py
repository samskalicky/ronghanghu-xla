from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob

import torch


def numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel


def consolidate_param(checkpoints, name, prefix, suffix):
    p_shard_list = []
    for rank, ckpt in enumerate(checkpoints):
        p_shard = ckpt["model"][name]
        p_shard_list.append(p_shard)

        shard_metadata = ckpt["shard_metadata"]["shard_info"][prefix]
        orig_name = shard_metadata[suffix]["_orig_name"]
        orig_size = shard_metadata[suffix]["_orig_size"]

    full_param = torch.cat(p_shard_list, dim=0)
    full_param = full_param[:numel(orig_size)].view(*orig_size)

    full_name = orig_name
    if prefix != "":
        full_name = prefix + "." + orig_name

    return full_param, full_name


def unflatten_param(p, metadata, prefix):
    param_names, param_shapes, param_numels = metadata    
    full_params = (t.view(s) for (t, s) in zip(p.split(param_numels), param_shapes))
    full_names = [n.replace("_fpw_module.", "") for n in param_names]
    if prefix != "":
        full_names = [prefix + "." + n for n in full_names]
    return full_params, full_names


def consolidate_and_unflatten(checkpoints):
    full_state_dict = OrderedDict()

    # consolidate the checkpoints
    world_size = len(checkpoints)

    for name, p in checkpoints[0]["model"].items():
        is_sharded = False
        name_splits = name.split(".")
        for idx, sep in enumerate(name_splits):
            if sep.startswith("_fsdp_shard"):
                is_sharded = True
                prefix = ".".join(name_splits[:idx])
                suffix = ".".join(name_splits[idx:])
                break

        if is_sharded:
            full_param, full_name = consolidate_param(checkpoints, name, prefix, suffix)
        else:
            # unsharded buffers (just use rank 0's state dict)
            full_param, full_name = p, name
        full_state_dict[full_name] = full_param

    flatten_info = checkpoints[0]["shard_metadata"]["flatten_info"]
    for name in list(full_state_dict):
        if "_fsdp_wrapped_module.flat_param_" in name:
            assert name in flatten_info
            p = full_state_dict.pop(name)
            metadata = flatten_info[name]
            prefix = ".".join(name.split(".")[:-1])
            full_params, full_names = unflatten_param(p, metadata, prefix)
            for fp, fn in zip(full_params, full_names):
                full_state_dict[fn] = fp

    full_state_dict = OrderedDict(
        (k.replace("_fsdp_wrapped_module.", ""), v)
        for k, v in full_state_dict.items()
    )

    return full_state_dict


def consolidate_xla_fsdp_model_state_dict(
    ckpt_prefix, ckpt_suffix="*.pth", save_path=""
):
    ckpt_files = glob(ckpt_prefix + ckpt_suffix)
    print(f"found {len(ckpt_files)} checkpoint files")
    assert len(ckpt_files) > 0
    checkpoints = [torch.load(f) for f in ckpt_files]
    checkpoints.sort(key=lambda c: c["shard_metadata"]["rank"])
    for rank, ckpt in enumerate(checkpoints):
        assert ckpt["shard_metadata"]["rank"] == rank
        assert ckpt["shard_metadata"]["world_size"] == len(checkpoints)

    full_state_dict = consolidate_and_unflatten(checkpoints)
    if save_path == "":
        save_path = ckpt_prefix + "_consolidated.pth"
    torch.save({"model": full_state_dict}, save_path)
    print(f"saved consolidated model to {save_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_prefix", type=str, required=True,
        help="the path prefix of the XLA FSDP checkpoints to be consolidated",
    )
    parser.add_argument(
        "--ckpt_suffix", type=str, default="_rank-*-of-*.pth",
        help="the path suffix pattern of the XLA FSDP checkpoints to be consolidated"
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="The save path of the consolidated checkpoint (default will be ckpt_prefix + '_consolidated.pth')",
    )
    args = parser.parse_args()
    consolidate_xla_fsdp_model_state_dict(
        args.ckpt_prefix, args.ckpt_suffix, args.save_path
    )


if __name__ == "__main__":
    main()
