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


def remote_report():
    """Ask HuggingFace how big each published quant is, without downloading any of it.

    File sizes come from the repo metadata, so this costs one API call per revision
    and settles a 60 GB download on whether the n-gram table would actually fit.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  huggingface_hub is not installed -- run this from the project venv:")
        print("    ~/.local/share/qwen38/venv/bin/python analyze-model.py --remote")
        return

    repo = "turboderp/Qwen3.8-Flash-Next-exl3"
    api = HfApi()
    ram = mem_total()
    try:
        refs = [b.name for b in api.list_repo_refs(repo).branches]
    except Exception as e:
        print(f"  could not list revisions: {e}")
        return

    print(f"  {'revision':22} {'total':>9} {'table':>9} {'arena':>9} {'spare':>9}   verdict")
    for rev in sorted(refs):
        try:
            info = api.model_info(repo, revision=rev, files_metadata=True)
        except Exception as e:
            print(f"  {rev:22}   (skipped: {type(e).__name__})")
            continue
        table = sum(f.size or 0 for f in info.siblings if "ngram" in f.rfilename)
        total = sum(f.size or 0 for f in info.siblings
                    if f.rfilename.endswith(".safetensors"))
        arena = total - table
        if not table or not arena:
            continue
        spare = ram - arena
        if table <= spare:
            verdict = f"FITS, {gb(spare - table):.1f} GB spare"
        else:
            verdict = f"{100 * (1 - spare / table):.0f}% of reads miss"
        print(f"  {rev:22} {gb(total):8.1f}G {gb(table):8.1f}G "
              f"{gb(arena):8.1f}G {gb(spare):8.1f}G   {verdict}")
    print()
    print("  'spare' is RAM left for page cache once the arena is resident. The OS needs")
    print("  a few GB of that, so anything under ~6 GB spare is a marginal fit, not a safe one.")


def main():
    p = argparse.ArgumentParser(description="Flash-Next -- does the n-gram table fit in RAM?")
    p.add_argument("-m", "--model_dir")
    p.add_argument("--remote", action="store_true",
                   help="compare every published quant by size; downloads nothing")
    p.add_argument("--table-gb", type=float, help="override: model the table at this size")
    p.add_argument("--arena-gb", type=float, help="override: model the arena at this size")
    args = p.parse_args()

    if args.remote:
        print("Flash-Next RAM budget -- published quants")
        print()
        remote_report()
        return
    if not args.model_dir:
        p.error("-m/--model_dir is required unless --remote is given")

    t = int(args.table_gb * 2 ** 30) if args.table_gb else None
    a = int(args.arena_gb * 2 ** 30) if args.arena_gb else None
    print("Flash-Next RAM budget")
    print()
    sys.exit(report(args.model_dir, a, t))


if __name__ == "__main__":
    main()