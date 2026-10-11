"""Shared loading and prompt helpers for the 8 GB scripts. MIT — Lna-Lab, 2026."""
import argparse, json, os

# FLASH_THINK=0 turns thinking off. Only has an effect if the chat template honours enable_thinking.
THINK = os.environ.get("FLASH_THINK", "1") != "0"


def add_common_args(parser):
    """The flags every script here takes. `-mcl` is the one that makes it fit."""
    from exllamav3 import model_init
    model_init.add_args(parser, add_draft_model_args=True)
    return parser


def load(args):
    """Load the model with routed experts on the CPU. Returns (model, config, cache, tokenizer)."""
    from exllamav3 import model_init
    if not getattr(args, "moe_cpu_offload", 0):
        args.moe_cpu_offload = 64          # every MoE layer's experts to system RAM
    res = model_init.init(args)
    return res[0], res[1], res[2], res[3]


def make_generator(model, cache, tokenizer):
    from exllamav3.generator import Generator
    return Generator(model=model, cache=cache, tokenizer=tokenizer)


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
