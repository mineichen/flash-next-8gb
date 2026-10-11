"""Shared loading and prompt helpers for the 8 GB scripts. MIT — Lna-Lab, 2026."""
import argparse, json, os

# FLASH_THINK=0 turns thinking off. Only has an effect if the chat template honours enable_thinking.
THINK = os.environ.get("FLASH_THINK", "1") != "0"

# KV cache size in tokens. Only the cache scales with context, and in this model it is unusually
# cheap: 36 of the 48 layers are linear attention with fixed-size recurrent state, so they cost the
# same at 1 token as at 100k. The 12 real attention layers use 2 KV heads at head_dim 256, which is
# ~24 KiB/token in fp16. Sized to leave room for the prefill scratch on a fully loaded 8 GB card;
# raise it with -cs once the startup report shows how much VRAM is actually free.
DEFAULT_CACHE_SIZE = int(os.environ.get("QWEN_CACHE_SIZE", 16384))

# Max tokens per prefill forward pass; every activation in that pass is sized by it. Lowering it is
# nearly free here, which is the point: the routed experts run on the CPU at ~30-60 ms/token, so a
# 512-token chunk is already ~30 s of work and kernel-launch overhead never shows up. At the
# ExLlamaV3 default of 2048 the PLE stream buffer alone is 40 MiB, and that is what OOMs a
# fully-loaded 8 GB card during prefill.
DEFAULT_CHUNK_SIZE = int(os.environ.get("QWEN_CHUNK_SIZE", 512))

# Off by default, and on ExLlamaV3 1.4.5 it cannot be turned on: this model has a QSA indexer, and
# 1.4.5's QSA cache layer is fp16-only (it asserts, in Attention.cache_layer_type). Master added
# CacheLayer_qsa_quant, which would halve this term. Leave empty until you are on a build that has it.
DEFAULT_CACHE_QUANT = os.environ.get("QWEN_CACHE_QUANT", "")


def add_common_args(parser):
    """The flags every script here takes. `-mcl` is the one that makes it fit."""
    import inspect
    from exllamav3 import model_init

    # ExLlamaV3 1.4.5 accepts default_cache_size but not default_chunk_size (that one is master-only),
    # so ask the function what it takes rather than what we happen to have read.
    supported = inspect.signature(model_init.add_args).parameters
    want = {"add_draft_model_args": True,
            "default_cache_size": DEFAULT_CACHE_SIZE,
            "default_chunk_size": DEFAULT_CHUNK_SIZE}
    model_init.add_args(parser, **{k: v for k, v in want.items() if k in supported})

    # Set these directly so they hold whether or not add_args grew a knob for them: make_generator reads
    # args.chunk_size, and this is what keeps prefill from falling back to ExLlamaV3's 2048.
    parser.set_defaults(cache_quant=DEFAULT_CACHE_QUANT, chunk_size=DEFAULT_CHUNK_SIZE)
    return parser


def _cache_quant(value):
    """`-cq 8,8` asks for a quantized cache; `-cq fp16` asks for none."""
    if value is None:
        return None
    v = str(value).strip().lower()
    return None if v in ("", "0", "none", "off", "fp16") else v


def load(args):
    """Load the model with routed experts on the CPU. Returns (model, config, cache, tokenizer)."""
    from exllamav3 import model_init
    if not getattr(args, "moe_cpu_offload", 0):
        args.moe_cpu_offload = 64          # every MoE layer's experts to system RAM
    args.cache_quant = _cache_quant(getattr(args, "cache_quant", None))
    try:
        res = model_init.init(args)
    except AssertionError as e:
        if "fp16 cache layer" in str(e):
            raise AssertionError(
                f"{e}\nThis model has a QSA (sparse) attention indexer, and the QSA cache layer is "
                f"fp16-only in this ExLlamaV3 build. Drop -cq (or set QWEN_CACHE_QUANT=). Newer builds "
                f"add CacheLayer_qsa_quant, which would halve the KV cache."
            ) from e
        raise
    return res[0], res[1], res[2], res[3]


def make_generator(model, cache, tokenizer, args=None):
    """Build the generator, forwarding the knobs the CLI already exposes.

    ExLlamaV3's Generator defaults to max_chunk_size 2048 and takes no CPU cache tier, so without
    this the `-cs`/`-ccs`/`-chunk_size` flags only half-reach the runtime: prefill chunks at 2048
    and allocates from the GPU whatever it should have kept in system RAM.
    """
    import inspect
    from exllamav3.generator import Generator

    want = {}
    if args is not None:
        want["max_chunk_size"] = getattr(args, "chunk_size", DEFAULT_CHUNK_SIZE)
        want["cpu_cache_size"] = int(float(getattr(args, "cpu_cache_size", 0) or 0) * 1024 ** 3)
        want["recurrent_cache_size"] = int(float(getattr(args, "recurrent_cache_size", 4.0) or 0) * 1024 ** 3)

    # Older ExLlamaV3 builds lack some of these; passing them anyway is a TypeError at startup.
    have = inspect.signature(Generator.__init__).parameters
    want = {k: v for k, v in want.items() if k in have}
    return Generator(model=model, cache=cache, tokenizer=tokenizer, **want)


def _child_pids(pid):
    """Children of pid, read straight out of procfs (no subprocess, no pgrep dependency)."""
    out = []
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for t in tasks:
        try:
            with open(f"/proc/{pid}/task/{t}/children") as f:
                out += [int(x) for x in f.read().split()]
        except (OSError, ValueError):
            pass
    return out


def hugepage_report():
    """Share of this process tree's memory in 2 MiB pages.

    The routed experts are ~58 GB of anonymous memory read in scattered order, and they
    live in a spawned child process rather than this one -- so the parent alone badly
    undercounts. At 4 KiB coverage essentially every read is a TLB miss, and decode
    runs an order of magnitude under what the memory bandwidth would allow.

    ExLlamaV3 asks for MADV_COLLAPSE once after load, but it is best-effort and gives
    up silently when memory is fragmented. So this is worth reading, not assuming.
    """
    pid = os.getpid()
    pids = [pid] + _child_pids(pid) + [c for p in _child_pids(pid) for c in _child_pids(p)]
    anon = huge = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Anonymous:"):
                        anon += int(line.split()[1])
                    elif line.startswith("AnonHugePages:"):
                        huge += int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
    if anon <= 0:
        return
    gib = 2 ** 30
    # Only meaningful once the model is actually resident; a few MiB of interpreter
    # anonymous memory would otherwise trip the warning on every single start.
    if anon < gib:
        return
    pct = huge * 100 // anon
    print(f"  hugepages  {huge / gib:.1f} of {anon / gib:.1f} GiB in 2 MiB pages  ({pct}%)",
          flush=True)
    if pct < 20:
        print(f"             ^^ only {pct}%. If decode is CPU-bound, this is why: scattered 4 KiB "
              f"reads over ~58 GB miss the TLB on almost every access.", flush=True)
        print("             Try, before starting the server:  sudo sysctl -w vm.drop_caches=3",
              flush=True)


def memory_report(cache, gen=None):
    """Print where VRAM actually went, so `-cs` can be chosen instead of guessed.

    Only the KV cache scales with context; everything else is fixed by the weights, so this is the
    one number that decides how long a conversation can get.
    """
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return
    mib = 1024 ** 2
    alloc, resv = torch.cuda.memory_allocated() / mib, torch.cuda.memory_reserved() / mib
    try:
        per_tok = sum(l.storage_size() for l in cache.layers.values()) / max(cache.max_num_tokens, 1)
    except Exception:
        per_tok = float("nan")
    kv = per_tok * cache.max_num_tokens / mib
    print(f"  VRAM      {alloc:8.0f} MiB allocated, {resv:.0f} reserved, "
          f"{free / mib:.0f} of {total / mib:.0f} MiB free", flush=True)
    print(f"  KV cache  {kv:8.0f} MiB   {per_tok / 1024:.1f} KiB/token, "
          f"{cache.max_num_tokens} tokens", flush=True)
    if gen is not None and hasattr(gen, "get_cache_stats"):
        s = gen.get_cache_stats()
        print(f"  pages     {s['used_tokens']} used, {s['free_tokens']} free, "
              f"hit rate {s['hit_rate']}", flush=True)
    hugepage_report()


def _fallback_prompt(messages):
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)


def _norm_messages(messages):
    """OpenAI request shape -> template shape: text parts joined, tool arguments as objects."""
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):                       # typed content parts, e.g. image/text blocks
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = dict(tc.get("function", tc))
                a = f.get("arguments", {})
                if isinstance(a, str):               # templates want the object, not JSON text
                    try:
                        a = json.loads(a or "{}")
                    except json.JSONDecodeError:
                        a = {}
                f["arguments"] = a
                tcs.append({**tc, "function": f})
            m["tool_calls"] = tcs
        out.append(m)
    return out


def render_chat(model_dir, messages, tools=None):
    """Render messages with the model's own chat template when we can, else ChatML."""
    messages = _norm_messages(messages)
    path = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.exists(path):
        return _fallback_prompt(messages)
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(msg):                    # Qwen templates call this on bad tool schemas
            raise Exception(msg)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, **kw: json.dumps(
            x, ensure_ascii=kw.get("ensure_ascii", False), indent=kw.get("indent"),
            separators=kw.get("separators"), sort_keys=kw.get("sort_keys", False))
        tpl = env.from_string(open(path, encoding="utf-8").read())
        return tpl.render(messages=messages, tools=tools or None,
                          add_generation_prompt=True, enable_thinking=THINK)
    except Exception as e:
        # Loudly. A silent ChatML fallback hides the reason the template was unusable.
        print(f"template render failed ({e}); falling back to plain ChatML", flush=True)
        return _fallback_prompt(messages)



def stop_conditions(tokenizer):
    """Where a reply ends. Without these the model runs on into <|endoftext|> and noise."""
    stops = ["<|im_end|>", "<|endoftext|>"]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stops.append(eos)
    return stops


def split_reasoning(text):
    """Flash-Next thinks before it answers. Return (reasoning, answer)."""
    for marker in ("</think>", "</thinking>"):
        if marker in text:
            head, _, tail = text.partition(marker)
            return head.strip(), tail.strip()
    return "", text.strip()
