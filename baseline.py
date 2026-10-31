#!/usr/bin/env python3
"""Baseline inference — Qwen3-4B zero-shot, compressed schema, ReAct loop.

Dùng gold audio_text (chưa ASR), đo năng lực LLM thuần.

Kaggle T4:
  !pip install transformers accelerate
  !python baseline.py --root /kaggle/input/lifemate --split train \
       --dev split_dev.json --out submission_baseline.jsonl --score
"""
from __future__ import annotations

import argparse, copy, json, re, sys, time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B"
MAX_STEPS = 8
MAX_GEN = 1024
MAX_TIME = 25
RESULT_CAP = 600

TYPE_MAP = {"string": "str", "integer": "int", "number": "num", "boolean": "bool"}


# ── compress schema ──────────────────────────────────────────────

def _compress_param(name, spec, required):
    typ = TYPE_MAP.get(spec.get("type", ""), spec.get("type", "?"))
    if spec.get("type") == "array":
        items = spec.get("items", {})
        if items.get("type") == "object":
            ip = items.get("properties", {})
            ir = set(items.get("required", []))
            fs = ",".join(
                f"{k}:{TYPE_MAP.get(v.get('type',''),'?')}{'*' if k in ir else ''}"
                for k, v in ip.items()
            )
            typ = "[{" + fs + "}]"
        else:
            typ = "[" + TYPE_MAP.get(items.get("type", ""), "?") + "]"

    s = f"{name}:{typ}"
    if name in required:
        s += "*"
    if "enum" in spec:
        s += "=" + "|".join(str(e) for e in spec["enum"])
    if "x-ref" in spec:
        s += "->" + spec["x-ref"]
    if spec.get("x-match") == "fuzzy":
        s += "~"
    if "exclusiveMinimum" in spec:
        s += ">" + str(spec["exclusiveMinimum"])
    elif "minimum" in spec:
        s += ">=" + str(spec["minimum"])
    return s


def compress_tool(tool):
    params = tool.get("parameters", {})
    props = params.get("properties", {})
    req = set(params.get("required", []))
    xoo = params.get("x-one-of", [])

    pstr = ", ".join(_compress_param(k, v, req) for k, v in props.items())
    line = f"{tool['name']}({pstr})"
    if xoo:
        line += " oneof:" + " ".join("/".join(g) for g in xoo)
    line += f"  # {tool.get('description', '')}"
    return line


def build_schema_text(catalog, installed_apps):
    allowed = set()
    for info in installed_apps.values():
        allowed.update(info["tools"])

    idx = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            idx[t["name"]] = (app["id"], t)

    by_app: dict[str, list] = {}
    for name in sorted(allowed):
        if name in idx:
            aid, spec = idx[name]
            by_app.setdefault(aid, []).append(spec)

    lines = []
    for aid in sorted(by_app):
        lines.append(f"[{aid}]")
        for t in by_app[aid]:
            lines.append(compress_tool(t))
    return "\n".join(lines)


# ── prompt ───────────────────────────────────────────────────────

_SYS = """Bạn là LifeMate, trợ lý AI. Thực hiện yêu cầu bằng tool.

Người dùng: {name} | Nhà: {home} | CQ: {work} | TP: {city}
Thời gian: {time} | Vị trí: {lat},{lng}

## Tool
Format: tên(tham_số)  # mô_tả
*=bắt buộc  =a|b=enum  ->ref=ID_tham_chiếu(cần đọc trước)  ~=văn_bản_tự_do  oneof:a/b=chọn_một

{schema}

## Cách trả lời
Gọi tool:
CALL: {{"name": "app.tool", "arguments": {{"key": "value"}}}}
Nhiều call độc lập → nhiều dòng CALL cùng lúc.
Xong hoặc ngoài phạm vi → DONE (không CALL).

## Quy tắc
1. CHỈ gọi tool trên. Ngoài phạm vi → DONE.
2. Cần ID → đọc trước (search/list/get_status). KHÔNG bịa ID.
3. Thiếu info bắt buộc không suy ra được → CALL sys.ask_missing_info.
4. Số nguyên không chấm/phẩy (1200000). Giờ: ISO 8601 +07:00.
5. Tối thiểu call cần thiết."""


def build_messages(item, catalog):
    ctx = item["context"]
    up = ctx["user_profile"]
    schema = build_schema_text(catalog, ctx["installed_apps"])

    msgs = [{"role": "system", "content": _SYS.format(
        name=up["name"], home=up["home_address"], work=up["work_address"],
        city=up["city"], time=ctx["current_time"],
        lat=ctx["location"]["lat"], lng=ctx["location"]["lng"],
        schema=schema,
    )}]

    for turn in (item.get("history") or []):
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": turn.get("audio_text", "")})
        elif turn["role"] == "assistant":
            msgs.append({"role": "assistant", "content": turn.get("text", "")})

    msgs.append({"role": "user", "content": item["audio_text"]})
    return msgs


# ── parse & execute ──────────────────────────────────────────────

def parse_calls(text):
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<\|[^|]*\|>", "", clean)  # strip special tokens
    calls = []
    for line in clean.split("\n"):
        line = line.strip()
        if line.startswith("CALL:"):
            s = line[5:].strip()
            s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing comma
            try:
                obj = json.loads(s)
                if "name" in obj:
                    calls.append({"name": obj["name"],
                                  "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass
    if calls:
        return calls
    # fallback: Qwen3 native <tool_call> tags
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", clean, re.DOTALL):
        s = re.sub(r",\s*([}\]])", r"\1", m.group(1))
        try:
            obj = json.loads(s)
            if "name" in obj:
                calls.append({"name": obj["name"],
                              "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError:
            pass
    return calls


def cap_result(result, limit=RESULT_CAP):
    r = copy.deepcopy(result) if isinstance(result, (dict, list)) else result
    if isinstance(r, dict):
        for k, v in r.items():
            if isinstance(v, list) and len(v) > 5:
                r[k] = v[:5] + [f"...({len(v) - 5} more)"]
    s = json.dumps(r, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 3] + "..."


# ── react loop ───────────────────────────────────────────────────

def react_loop(gen_fn, messages, env):
    t0 = time.time()
    for _ in range(MAX_STEPS):
        if time.time() - t0 > MAX_TIME:
            break

        resp = gen_fn(messages)
        calls = parse_calls(resp)
        if not calls:
            break

        messages.append({"role": "assistant", "content": resp})

        parts = []
        for c in calls:
            val = env.validate(c["name"], c.get("arguments", {}))
            if val.get("ok", False):
                res = env.call(c["name"], c.get("arguments", {}))
            else:
                res = val
            parts.append(f"{c['name']} → {cap_result(res)}")

        messages.append({
            "role": "user",
            "content": "[Kết quả tool]\n" + "\n".join(parts) + "\n\nTiếp tục.",
        })

    return env.get_tool_calls(), env.get_state_hash()


# ── model ────────────────────────────────────────────────────────

def _apply(tok, msgs, think):
    kw = {"add_generation_prompt": True, "tokenize": False}
    if think is not None:
        try:
            return tok.apply_chat_template(msgs, enable_thinking=think, **kw)
        except TypeError:
            pass
    return tok.apply_chat_template(msgs, **kw)


def load_model(model_id, think=True):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {model_id} ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    def generate(messages):
        text = _apply(tok, messages, think)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=15000)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_GEN, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=False)

    return generate


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", type=Path, help="split_dev.json — chỉ chạy dev set")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("submission_baseline.jsonl"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--score", action="store_true", help="chấm điểm sau khi xong")
    a = ap.parse_args()

    catalog = json.loads((a.root / a.split / f"{a.split}_tools_catalog.json")
                         .read_text(encoding="utf-8"))

    dev_ids = None
    if a.dev:
        dev_ids = set(json.loads(a.dev.read_text(encoding="utf-8"))["dev"])

    samples = []
    with open(a.root / a.split / f"{a.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if dev_ids is None or d["id"] in dev_ids:
                    samples.append(d)
    if a.limit:
        samples = samples[: a.limit]
    print(f"{len(samples)} samples loaded")

    gen = load_model(a.model, think=not a.no_think)

    sys.path.insert(0, str(a.root / a.split))
    from zaic_tools import ToolEnv

    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as fout:
        for i, item in enumerate(samples):
            ts = time.time()
            env = ToolEnv(a.split, item["id"], root=a.root)
            msgs = build_messages(item, catalog)
            tc, sh = react_loop(gen, msgs, env)

            fout.write(json.dumps({"id": item["id"], "tool_calls": tc,
                                   "state_hash": sh}, ensure_ascii=False) + "\n")
            print(f"  [{i + 1}/{len(samples)}] {item['id']}: "
                  f"{len(tc)} calls, {time.time() - ts:.1f}s")

    total = time.time() - t0
    print(f"\n{len(samples)} mẫu — {total:.0f}s "
          f"({total / max(len(samples), 1):.1f}s/mẫu) → {a.out}")

    if a.score and "label" in samples[0]:
        import subprocess
        cmd = [sys.executable, str(Path(__file__).parent / "score.py"),
               "--split", a.split, "--root", str(a.root), "--sub", str(a.out)]
        if a.dev:
            cmd += ["--dev", str(a.dev)]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
