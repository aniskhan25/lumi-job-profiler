#!/usr/bin/env python3
"""Minimal PyTorch ROCm demo workload.

Runs a sustained GEMM loop on GPU to generate utilization and memory activity.
"""

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ROCm demo workload")
    parser.add_argument("--seconds", type=int, default=30, help="Runtime duration")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (NxN)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Compute dtype")
    parser.add_argument("--device", default="cuda:0", help="Device string (default: cuda:0)")
    parser.add_argument("--reserve-mem-gb", type=float, default=0.0, help="Optional GPU memory to reserve")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    return parser.parse_args()


def dtype_from_str(name):
    import torch
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def reserve_memory(device, dtype, gb):
    import torch
    if gb <= 0:
        return None
    bytes_per_elem = torch.tensor([], device=device, dtype=dtype).element_size()
    num_elems = int((gb * (1024 ** 3)) / bytes_per_elem)
    if num_elems <= 0:
        return None
    return torch.empty(num_elems, device=device, dtype=dtype)


def main():
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch not available: {exc}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False. ROCm/CUDA not available.")
        sys.exit(1)

    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)

    torch.manual_seed(0)
    # torch.cuda.set_device expects an index or a device with index.
    if device.index is None:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    print("Device:", props.name)
    print("Total memory (GB):", round(props.total_memory / (1024 ** 3), 2))
    if torch.version.hip:
        print("ROCm version:", torch.version.hip)

    reserve = reserve_memory(device, dtype, args.reserve_mem_gb)
    if reserve is not None:
        print(f"Reserved approx {args.reserve_mem_gb} GB on GPU")

    n = args.size
    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)

    for _ in range(args.warmup):
        c = a @ b
        torch.cuda.synchronize()

    print("Starting compute loop...")
    start = time.time()
    iters = 0
    last_log = start

    while time.time() - start < args.seconds:
        c = a @ b
        if dtype != torch.float32:
            c = c.float().sum()
        torch.cuda.synchronize()
        iters += 1

        if iters % args.log_interval == 0:
            now = time.time()
            dt = now - last_log
            last_log = now
            print(f"iter={iters} elapsed={int(now - start)}s interval={dt:.2f}s")

    elapsed = time.time() - start
    print(f"Done. iters={iters} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
