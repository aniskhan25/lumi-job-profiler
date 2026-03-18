#!/usr/bin/env python3
"""Minimal distributed PyTorch ROCm demo workload for LUMI.

The script expects to be launched under Slurm with one rank per GPU.
It initializes torch.distributed, runs small GEMMs on the local GPU,
and periodically executes an all-reduce to generate distributed activity.
"""

import argparse
import os
import socket
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed PyTorch ROCm demo workload")
    parser.add_argument("--seconds", type=int, default=30, help="Runtime duration")
    parser.add_argument("--size", type=int, default=2048, help="Matrix size (NxN)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Compute dtype")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--sync-interval", type=int, default=5, help="Run all-reduce every N iterations")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations on rank 0")
    return parser.parse_args()


def dtype_from_str(name):
    import torch

    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def get_slurm_int(name, default=None):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def main():
    args = parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        print(f"PyTorch distributed not available: {exc}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False. ROCm/CUDA not available.")
        sys.exit(1)

    rank = get_slurm_int("SLURM_PROCID", 0)
    local_rank = get_slurm_int("SLURM_LOCALID", 0)
    world_size = get_slurm_int("SLURM_NTASKS", 1)
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = dtype_from_str(args.dtype)

    backend = "nccl"
    dist.init_process_group(backend=backend, init_method="env://")

    props = torch.cuda.get_device_properties(device)
    if rank == 0:
        print(f"world_size={world_size} backend={backend}")
    print(
        f"rank={rank} local_rank={local_rank} host={hostname} device={props.name} total_mem_gb={props.total_memory / (1024 ** 3):.2f}",
        flush=True,
    )

    torch.manual_seed(rank)
    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)
    sync_tensor = torch.tensor([rank + 1.0], device=device)

    for _ in range(args.warmup):
        c = a @ b
        sync_tensor += c.float().mean()
        dist.all_reduce(sync_tensor)
        torch.cuda.synchronize()

    start = time.time()
    iters = 0
    last_log = start

    while time.time() - start < args.seconds:
        c = a @ b
        if args.sync_interval > 0 and iters % args.sync_interval == 0:
            sync_tensor.copy_(c.float().mean().reshape(1))
            dist.all_reduce(sync_tensor)
        torch.cuda.synchronize()
        iters += 1

        if rank == 0 and iters % args.log_interval == 0:
            now = time.time()
            print(
                f"iter={iters} elapsed={int(now - start)}s interval={now - last_log:.2f}s reduced={float(sync_tensor.item()):.4f}",
                flush=True,
            )
            last_log = now

    dist.barrier()
    elapsed = time.time() - start
    if rank == 0:
        print(f"done iters={iters} elapsed={elapsed:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
