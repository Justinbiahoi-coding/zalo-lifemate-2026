#!/usr/bin/env python3
"""Convert gold labels → multi-turn SFT data for Qwen3.

Groups gold tool_calls into ReAct steps by dependency analysis:
  - Level 0: calls whose args don't need previous results (reads, queries)
  - Level 1+: calls that use IDs/values obtained from previous results

Output: sft_train.jsonl — {"messages": [...]} per line, ready for Unsloth QLoRA.

    python make_sft.py                        # default: all train-split samples
    python make_sft.py --dev split_dev.json   # only train portion of dev split
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path
from baseline import (compress_tool, build_schema_text, _SYS, cap_result,
                      fmt_call_lines, default_root)


def _build_xref(catalog):
    out = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            props = (t.get("parameters") or {}).get("properties") or {}
            refs = {k for k, v in props.items() if "x-ref" in v}
            if refs:
                out[t["name"]] = refs
    return out


def _build_fuzzy(catalog):
    out = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            props = (t.get("parameters") or {}).get("properties") or {}
            fz = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
            if fz:
                out[t["name"]] = fz
    return out


def _extract_strings(obj, out):
    if isinstance(obj, str) and len(obj) >= 3:
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _extract_strings(item, out)


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def group_calls(tool_calls, tool_results, xref_map, fuzzy_map):
    n = len(tool_calls)
    if n == 0:
        return []

    result_vals = []
    for r in tool_results:
        s = set()
        _extract_strings(r, s)
        result_vals.append(s)

    levels = []
    for i, call in enumerate(tool_calls):
        name = call["name"]
        args = call.get("arguments", {})
        xrefs = xref_map.get(name, set())
        fuzzys = fuzzy_map.get(name, set())
        max_dep = -1

        for k, v in args.items():
            if k in fuzzys:
                continue
            if not isinstance(v, str) or len(v) < 3:
                continue
            if _ISO.match(v):
                continue
            is_xref = k in xrefs
            for j in range(i):
                if v in result_vals[j]:
                    max_dep = max(max_dep, levels[j])
                    break
            else:
                if is_xref:
                    max_dep = max(max_dep, 0)

        levels.append(max_dep + 1)

    mx = max(levels)
    groups = []
    for lv in range(mx + 1):
        g = [(tool_calls[i], tool_results[i]) for i in range(n) if levels[i] == lv]
        if g:
            groups.append(g)
    return groups


def _fmt_calls(pairs):
    return fmt_call_lines([c for c, _ in pairs])


def _fmt_results(pairs):
    parts = []
    for c, r in pairs:
        parts.append(f"{c['name']} → {cap_result(r)}")
    return "[Kết quả tool]\n" + "\n".join(parts) + "\n\nTiếp tục."


def sample_to_messages(item, catalog, xref_map, fuzzy_map, done_frac=0.3, rng=None):
    ctx = item["context"]
    up = ctx["user_profile"]
    schema = build_schema_text(catalog, ctx["installed_apps"])

    msgs = [{"role": "system", "content": _SYS.format(
        name=up["name"], home=up["home_address"], work=up["work_address"],
        city=up["city"], time=ctx["current_time"],
        lat=ctx["location"]["lat"], lng=ctx["location"]["lng"],
        schema=schema,
    )}]

    for turn in item.get("history") or []:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": turn.get("audio_text", "")})
        elif turn["role"] == "assistant":
            msgs.append({"role": "assistant",
                         "content": fmt_call_lines(turn.get("tool_calls") or [])})

    from vn_normalize import normalize as vn_norm
    user_text = vn_norm(item["audio_text"], ctx.get("current_time"))
    msgs.append({"role": "user", "content": user_text})

    tc = item["label"]["tool_calls"]
    tr = item["label"]["tool_results"]

    if not tc:
        msgs.append({"role": "assistant", "content": "DONE"})
        return msgs

    groups = group_calls(tc, tr, xref_map, fuzzy_map)
    for i, grp in enumerate(groups):
        msgs.append({"role": "assistant", "content": _fmt_calls(grp)})
        if i < len(groups) - 1:
            msgs.append({"role": "user", "content": _fmt_results(grp)})

    # Terminal "stop" turn. Single-group samples outnumber multi-group ~2:1, so
    # adding it everywhere biases greedy decoding to stop after the first results.
    if len(groups) > 1 or (rng and rng.random() < done_frac):
        msgs.append({"role": "user", "content": _fmt_results(groups[-1])})
        msgs.append({"role": "assistant", "content": "DONE"})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dev", type=Path, help="split_dev.json — chỉ lấy train portion")
    ap.add_argument("--out", type=Path, default=Path("sft_train.jsonl"))
    ap.add_argument("--done-frac", type=float, default=0.3,
                    help="fraction of single-group samples that get a terminal DONE turn")
    a = ap.parse_args()
    if a.root is None:
        a.root = default_root()
    print(f"root = {a.root}")

    import random
    rng = random.Random(42)

    catalog = json.loads(
        (a.root / a.split / f"{a.split}_tools_catalog.json").read_text("utf-8")
    )
    xref_map = _build_xref(catalog)
    fuzzy_map = _build_fuzzy(catalog)

    ids_filter = None
    if a.dev:
        ids_filter = set(json.loads(a.dev.read_text("utf-8"))["train"])

    samples = []
    with open(a.root / a.split / f"{a.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if ids_filter is None or d["id"] in ids_filter:
                samples.append(d)

    print(f"{len(samples)} samples")

    oos_repeat = 3
    stats = {"total": 0, "oos": 0, "groups": {}, "stop": 0, "cont": 0}
    with open(a.out, "w", encoding="utf-8") as fout:
        for s in samples:
            msgs = sample_to_messages(s, catalog, xref_map, fuzzy_map,
                                      done_frac=a.done_frac, rng=rng)
            tc = s["label"]["tool_calls"]
            is_oos = not tc
            reps = oos_repeat if is_oos else 1
            for _ in range(reps):
                fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                stats["total"] += 1
            if is_oos:
                stats["oos"] += 1
            else:
                tr = s["label"]["tool_results"]
                ng = len(group_calls(tc, tr, xref_map, fuzzy_map))
                stats["groups"][ng] = stats["groups"].get(ng, 0) + 1
                stats["cont"] += ng - 1
                stats["stop"] += msgs[-1]["content"] == "DONE"

    print(f"→ {a.out}  ({stats['total']} conversations, OOS×{oos_repeat})")
    print(f"  OOS: {stats['oos']} (×{oos_repeat} = {stats['oos']*oos_repeat} in data)")
    print(f"  Groups distribution: {dict(sorted(stats['groups'].items()))}")
    print(f"  Termination balance: stop={stats['stop']} continue={stats['cont']} "
          f"({stats['stop']/max(stats['cont'],1):.2f}:1)")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        lens = []
        with open(a.out) as f:
            for line in f:
                conv = json.loads(line)
                text = tok.apply_chat_template(conv["messages"], tokenize=False)
                lens.append(len(tok.encode(text)))
        lens.sort()
        print(f"  Token lengths: median={lens[len(lens)//2]}, "
              f"p95={lens[int(len(lens)*0.95)]}, max={lens[-1]}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
