#!/usr/bin/env python3
"""Does the n-gram table fit in RAM, or is decode going to hit the disk?

This model does not fit in 64 GB of RAM and that is not a tuning problem -- it is
arithmetic. The routed experts (~47 GB) must stay resident, because a token reads
10 experts in each of 48 layers and evicting any of them puts ~2 GB/token on the
disk instead of the network. The n-gram table is the other ~33 GB, and it is mmapped
and served through the page cache, which is already an LRU cache -- so the only
question is whether it fits, and the only lever is making it smaller.

    python analyze-model.py -m ~/.local/share/qwen38/model

MIT -- Lna-Lab, 2026. Stdlib only.
"""
import argparse, os, sys

TABLE_FILE = "ngram_embedding.safetensors"


def dir_size(path, skip=()):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f in skip:
                continue
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def mem_total():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def gb(n):
    return n / 2 ** 30


def report(model_dir, arena=None, table=None):
    tb = os.path.join(model_dir, TABLE_FILE)
    on_disk = dir_size(model_dir)

    if table is None:
        try:
            table = os.stat(tb).st_size
        except OSError:
            print(f"!! {tb} not found. Point -m at the model directory.", file=sys.stderr)
            return 1
    if arena is None:
        arena = max(on_disk - table, 0)

    ram = mem_total()
    # Leave the OS and a little slack; page cache is only what is left over.
    spare = ram - arena
    fits = table <= spare
    cached_pct = min(100.0, spare / table * 100) if table else 0

    print(f"  model on disk          {gb(on_disk):6.1f} GB")
    print(f"  n-gram table           {gb(table):6.1f} GB   {TABLE_FILE}")
    print(f"  arena (must be in RAM) {gb(arena):6.1f} GB")
    print(f"  system RAM             {gb(ram):6.1f} GB")
    print()
    print(f"  RAM left for the table {gb(spare):6.1f} GB")
    print(f"  best possible cache    {cached_pct:6.1f}% of the table")
    print()
    if fits:
        print("  FITS. The whole table can sit in page cache, so no token needs the disk.")
        print("  Decode becomes memory-bound. This is the configuration you want.")
    else:
        miss = 1 - spare / table
        print(f"  DOES NOT FIT. At least {miss * 100:.0f}% of every token's table reads miss")
        print("  page cache and go to the SSD, and decode is disk-bound -- no CPU-side")
        print("  setting changes that.")
        print()
        print("  Two ways to close the gap, in order of preference:")
        print("    1. A smaller quant. The '2.05bpw_h4_ng4' build has the same model at 4-bit")
        print("       n-gram instead of 5-bit, so both the table and the arena shrink.")
        print("       Check it before downloading:")
        print(f"         python {sys.argv[0]} -m {model_dir} --table-gb 26 --arena-gb 32")
        print("    2. More RAM, then run with -ngr to load the table outright.")
    return 0


def main():
    p = argparse.ArgumentParser(description="Flash-Next -- does the n-gram table fit in RAM?")
    p.add_argument("-m", "--model_dir", required=True)
    p.add_argument("--table-gb", type=float, help="override: model the table at this size")
    p.add_argument("--arena-gb", type=float, help="override: model the arena at this size")
    args = p.parse_args()
    t = int(args.table_gb * 2 ** 30) if args.table_gb else None
    a = int(args.arena_gb * 2 ** 30) if args.arena_gb else None
    print("Flash-Next RAM budget")
    print()
    sys.exit(report(args.model_dir, a, t))


if __name__ == "__main__":
    main()