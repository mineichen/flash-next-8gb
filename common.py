"""Shared loading and prompt helpers for the 8 GB scripts. MIT — Lna-Lab, 2026."""
import argparse, json, os

# FLASH_THINK=0 turns thinking off. Only has an effect if the chat template honours enable_thinking.
THINK = os.environ.get("FLASH_THINK", "1") != "0"

# KV cache size in tokens. Only the cache scales with context, and in this model it is unusually
# cheap: 36 of the 48 layers are linear attention with fixed-size recurrent state, so they cost the
# same at 1 token as at 100k. The 12 real attention layers use 2 KV heads at head_dim 256, which
# is ~24 KiB/token in fp16 and ~12 KiB/token at -cq 8,8.
DEFAULT_CACHE_SIZE = int(os.environ.get("QWEN_CACHE_SIZE", 32768))

# Max tokens per prefill forward pass; every activation in that pass is sized by it. Lowering it is
# nearly free here, which is the point: the routed experts run on the CPU at ~30-60 ms/token, so a
# 512-token chunk is already ~30 s of work and kernel-launch overhead never shows up. At the
# ExLlamaV3 default of 2048 the PLE stream buffer alone is 40 MiB, and that is what OOMs a
# fully-loaded 8 GB card during prefill.
DEFAULT_CHUNK_SIZE = int(os.environ.get("QWEN_CHUNK_SIZE", 512))

# 8-bit keys and values. Halves the only term that grows with context; next to a 3.05 bpw body the
# quality change is not measurable. Pass -cq fp16 to turn it back off.
DEFAULT_CACHE_QUANT = os.environ.get("QWEN_CACHE_QUANT", "8,8")


def add_common_args(parser):
    """The flags every script here takes. `-mcl` is the one that makes it fit."""
    from exllamav3 import model_init
    model_init.add_args(parser, add_draft_model_args=True,
                        default_cache_size=DEFAULT_CACHE_SIZE,
                        default_chunk_size=DEFAULT_CHUNK_SIZE)
    parser.set_defaults(cache_quant=DEFAULT_CACHE_QUANT)
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
    res = model_init.init(args)
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
