#!/usr/bin/env python3
"""A small OpenAI-compatible server, so anything that speaks that API can use the model.
MIT — Lna-Lab, 2026.

    python serve.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw --port 8080

Stdlib only — no FastAPI, no uvicorn. One request at a time, which is the right shape here: the
experts run on the CPU, so a second concurrent stream does not get you more tokens per second.

    curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \\
      -d '{"model":"flash-next","messages":[{"role":"user","content":"hello"}],"stream":true}'
"""
import argparse, contextlib, json, os, re, threading, time, traceback, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A fully loaded 8 GB card has no slack: exllamav3 and torch fight over the last few tens of MiB during
# prefill, and the caching allocator's default segments make that worse. Set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import common

STATE = {}
LOCK = threading.Lock()
BUSY_TIMEOUT = float(os.environ.get("SERVE_BUSY_TIMEOUT", "120"))

# Qwen emits tool calls as XML; the API wants JSON objects.
TC_OPEN = "<tool_call>"
TC_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    """Qwen XML tool calls -> OpenAI tool_calls. Returns (calls, text_without_the_calls)."""
    schemas = {}
    for t in tools or []:
        f = t.get("function", {})
        schemas[f.get("name")] = f.get("parameters", {}).get("properties", {})
    calls = []
    for name, body in TC_RE.findall(text):
        props = schemas.get(name, {})
        args = {}
        for k, v in PARAM_RE.findall(body):
            if props.get(k, {}).get("type") != "string":   # the template quotes strings, not numbers
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    return calls, TC_RE.sub("", text).strip()


class Busy(Exception):
    """Another request already owns the model for longer than we are willing to queue."""


class OutOfMemory(Exception):
    """The prefill did not fit. The generator is reset before this escapes."""


class ContextTooLong(Exception):
    """Prompt plus the reserved generation budget does not fit the KV cache."""


@contextlib.contextmanager
def gpu_slot(timeout=BUSY_TIMEOUT):
    """One request at a time. Queue briefly, then tell the client to come back rather than hang."""
    if not LOCK.acquire(timeout=timeout):
        raise Busy()
    try:
        yield
    finally:
        LOCK.release()


def recover():
    """A job that died mid-prefill can leave pages, recurrent state and queued work behind. Drop it all.

    Without this the failure is sticky: the next request retries the dead job, allocates on top of the
    stranded memory, and OOMs at exactly the same byte count, forever.
    """
    gen = STATE.get("gen")
    if gen is not None:
        try:
            gen.clear_queue()
        except Exception:
            traceback.print_exc()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate(messages, max_new, tools=None, stream_cb=None):
    from exllamav3.generator import Job
    gen, tokenizer, model_dir = STATE["gen"], STATE["tokenizer"], STATE["model_dir"]
    prompt = common.render_chat(model_dir, messages, tools)
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=max_new,
              stop_conditions=common.stop_conditions(tokenizer))
    try:
        gen.enqueue(job)
    except AssertionError as e:
        # ExLlamaV3's page-table admission check. The cache has to hold the prompt AND the whole
        # reserved generation budget at the same time, so this fires whenever prompt + max_tokens
        # exceeds -cs -- which a client hits by trusting `limit.context` as a total rather than as
        # a prompt budget. Say that, instead of leaking the bare assertion.
        pin = ids.shape[-1]
        raise ContextTooLong(
            f"{e} — this request is {pin} prompt tokens plus {max_new} reserved for the reply, "
            f"so -cs must be at least {pin + max_new}. Raise -cs, or lower the client's max_tokens."
        ) from e
    out = []
    try:
        done = False
        while not done:
            for r in gen.iterate():
                if r.get("job") is not job:       # someone else's round, not ours to route
                    continue
                chunk = r.get("text", "")
                if chunk:
                    out.append(chunk)
                    if stream_cb:
                        stream_cb(chunk)
                err = r.get("error")
                if err is not None:
                    raise err
                if r.get("eos"):
                    done = True
                    break
        return "".join(out), ids.shape[-1], getattr(job, "new_tokens", len(out))
    except Exception as e:
        if "out of memory" in str(e).lower():
            recover()
            raise OutOfMemory(f"the prompt did not fit in GPU memory ({len(prompt)} chars)") from e
        raise
    finally:
        try:
            gen.cancel(job)                      # releases pages/state; a no-op once the job has finished
        except Exception:
            traceback.print_exc()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            return self._json(200, {"object": "list", "data": [
                {"id": STATE["alias"], "object": "model", "owned_by": "lna-lab"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})

        messages = req.get("messages") or []
        tools = req.get("tools") or None
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 1024)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if not req.get("stream"):
            try:
                with gpu_slot():
                    text, pin, pout = generate(messages, max_new, tools)
            except Busy:
                return self._json(503, {"error": {"message": "server busy, retry shortly",
                                                  "type": "server_busy"}})
            except ContextTooLong as e:
                return self._json(400, {"error": {"message": str(e), "type": "context_too_long"}})
            except OutOfMemory as e:
                return self._json(503, {"error": {"message": str(e), "type": "out_of_memory"}})
            reasoning, answer = common.split_reasoning(text)
            calls, answer = parse_tool_calls(answer, tools)
            msg = {"role": "assistant", "content": answer}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            finish = "tool_calls" if calls else ("length" if pout >= max_new else "stop")
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created,
                "model": STATE["alias"],
                "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
                "usage": {"prompt_tokens": pin, "completion_tokens": pout,
                          "total_tokens": pin + pout}})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def frame(delta, finish=None):
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": STATE["alias"],
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

        # Flash-Next thinks first. Route the thinking to reasoning_content, the answer to content,
        # holding a few characters back so a "</think>" split across chunks is still recognised.
        st = {"thinking": common.THINK, "buf": "", "hold": "", "tool": False, "tool_text": ""}

        def emit_content(chunk):
            # Forward answer text, but stop the moment a tool call opens — parse it whole at the end.
            if st["tool"]:
                st["tool_text"] += chunk
                return
            st["hold"] += chunk
            i = st["hold"].find(TC_OPEN)
            if i >= 0:
                if st["hold"][:i]:
                    frame({"content": st["hold"][:i]})
                st["tool"], st["tool_text"], st["hold"] = True, st["hold"][i:], ""
                return
            if len(st["hold"]) > len(TC_OPEN):          # keep back enough to spot an opener
                frame({"content": st["hold"][:-len(TC_OPEN)]})
                st["hold"] = st["hold"][-len(TC_OPEN):]

        def route(chunk):
            if not st["thinking"]:
                return emit_content(chunk)
            st["buf"] += chunk
            if "</think>" in st["buf"]:
                head, _, tail = st["buf"].partition("</think>")
                if head.strip():
                    frame({"reasoning_content": head})
                st["thinking"], st["buf"] = False, ""
                if tail.strip():
                    emit_content(tail)
            elif len(st["buf"]) > 4000:          # no marker in sight; treat it as the answer
                buf, st["buf"], st["thinking"] = st["buf"], "", False
                emit_content(buf)
            elif len(st["buf"]) > 64:
                frame({"reasoning_content": st["buf"][:-8]})
                st["buf"] = st["buf"][-8:]

        try:
            frame({"role": "assistant", "content": ""})
            with gpu_slot():
                text, pin, pout = generate(messages, max_new, tools, stream_cb=route)
            if st["thinking"] and st["buf"].strip():
                frame({"reasoning_content": st["buf"]})
            if not st["tool"] and st["hold"]:
                frame({"content": st["hold"]})
            calls = []
            if st["tool"]:
                calls, rest = parse_tool_calls(st["tool_text"], tools)
                if rest:
                    frame({"content": rest})
            for i, c in enumerate(calls):
                frame({"tool_calls": [{"index": i, **c}]})
            frame({}, finish="tool_calls" if calls else "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (Busy, OutOfMemory, ContextTooLong) as e:
            self._stream_error(cid, created, type(e).__name__, str(e))
        except Exception as e:
            traceback.print_exc()
            self._stream_error(cid, created, type(e).__name__, str(e))

    def _stream_error(self, cid, created, kind, msg):
        """Headers are already out, so the failure goes down the stream instead of as a status code."""
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": STATE["alias"], "error": {"message": msg[:500], "type": kind}}
        try:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    p = argparse.ArgumentParser(description="Flash-Next on 8 GB — OpenAI-compatible server")
    common.add_common_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--alias", default="flash-next")
    args = p.parse_args()

    model, config, cache, tokenizer = common.load(args)
    STATE.update(gen=common.make_generator(model, cache, tokenizer, args),
                 tokenizer=tokenizer, model_dir=args.model_dir, alias=args.alias)
    STATE["gen"].generate("Hello", max_new_tokens=8)
    common.memory_report(cache, STATE["gen"])
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}  (model: {args.alias})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()