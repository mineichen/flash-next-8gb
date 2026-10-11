#!/usr/bin/env python3
"""A small OpenAI-compatible server, so anything that speaks that API can use the model.
MIT — Lna-Lab, 2026.

    python serve.py -m ./Qwen3.8-Flash-Next-exl3-3.05bpw --port 8080

Stdlib only — no FastAPI, no uvicorn. One request at a time, which is the right shape here: the
experts run on the CPU, so a second concurrent stream does not get you more tokens per second.

    curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \\
      -d '{"model":"flash-next","messages":[{"role":"user","content":"hello"}],"stream":true}'
"""
import argparse, json, re, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import common

STATE = {}
LOCK = threading.Lock()

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


def generate(messages, max_new, tools=None, stream_cb=None):
    from exllamav3.generator import Job
    gen, tokenizer, model_dir = STATE["gen"], STATE["tokenizer"], STATE["model_dir"]
    prompt = common.render_chat(model_dir, messages, tools)
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=max_new,
              stop_conditions=common.stop_conditions(tokenizer))
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            chunk = r.get("text", "")
            if chunk:
                out.append(chunk)
                if stream_cb:
                    stream_cb(chunk)
    return "".join(out), ids.shape[-1], getattr(job, "new_tokens", len(out))


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
            with LOCK:
                text, pin, pout = generate(messages, max_new, tools)
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
            with LOCK:
                generate(messages, max_new, tools, stream_cb=route)
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


def main():
    p = argparse.ArgumentParser(description="Flash-Next on 8 GB — OpenAI-compatible server")
    common.add_common_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--alias", default="flash-next")
    args = p.parse_args()

    model, config, cache, tokenizer = common.load(args)
    STATE.update(gen=common.make_generator(model, cache, tokenizer),
                 tokenizer=tokenizer, model_dir=args.model_dir, alias=args.alias)
    STATE["gen"].generate("Hello", max_new_tokens=8)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}  (model: {args.alias})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()