"""Bộ chấm điểm local cho LifeMate, tái hiện metric của BTC.

Điểm mỗi mẫu:
  - gold rỗng  -> 1 nếu bài nộp cũng rỗng, ngược lại 0
  - gate       -> replay tool_calls đã nộp, state_hash phải bằng hash của gold; sai -> 0
  - qua gate   -> F1 giữa tập call đã nộp và gold

Khớp 1 call = cùng tên tool + mọi tham số non-fuzzy giống hệt + mọi tham số fuzzy
có token-F1 > 0.6. Thứ tự không tính, mỗi call khớp tối đa 1 call (ghép cặp tối ưu).

    python score.py --split train --sub submission.jsonl
    python score.py --split train --self-check      # chấm gold bằng gold, phải ra 1.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

FUZZY_THRESHOLD = 0.6


# ---------------------------------------------------------------- khớp tham số

def _tokens(s) -> set[str]:
    """Token của một chuỗi sau khi NFC + lower, chỉ giữ chữ và số."""
    return set(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(s)).lower()))


def token_f1(a, b) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    hit = len(ta & tb)
    if not hit:
        return 0.0
    p, r = hit / len(ta), hit / len(tb)
    return 2 * p * r / (p + r)


def load_fuzzy_params(catalog_path: Path) -> dict[str, set[str]]:
    """{tên tool: {tham số có x-match == "fuzzy"}} đọc từ catalog của split."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools = catalog["tools"] if "tools" in catalog else [t for a in catalog["apps"] for t in a.get("tools", [])]
    out = {}
    for t in tools:
        props = (t.get("parameters") or {}).get("properties") or {}
        out[t["name"]] = {k for k, v in props.items() if isinstance(v, dict) and v.get("x-match") == "fuzzy"}
    return out


def calls_match(sub: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    name = gold.get("name")
    if sub.get("name") != name:
        return False
    fz = fuzzy.get(name, set())
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if {k: v for k, v in sa.items() if k not in fz} != {k: v for k, v in ga.items() if k not in fz}:
        return False
    for k in set(sa) & set(ga) & fz:
        if token_f1(sa[k], ga[k]) <= FUZZY_THRESHOLD:
            return False
    return set(sa) & fz == set(ga) & fz  # cùng tập tham số fuzzy được truyền


def max_matching(subs: list[dict], golds: list[dict], fuzzy) -> int:
    """Ghép cặp tối đa (Kuhn) — greedy có thể đếm thiếu khi một call khớp nhiều gold."""
    adj = [[j for j, g in enumerate(golds) if calls_match(s, g, fuzzy)] for s in subs]
    pair = [-1] * len(golds)

    def try_assign(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if pair[j] == -1 or try_assign(pair[j], seen):
                pair[j] = i
                return True
        return False

    return sum(try_assign(i, set()) for i in range(len(subs)))


# ---------------------------------------------------------------- chấm điểm

def score_sample(env_factory, sample_id, sub_calls, gold_calls, fuzzy) -> dict:
    """Trả về dict chi tiết cho một mẫu. env_factory(sample_id) -> ToolEnv mới."""
    if not gold_calls:
        ok = not sub_calls
        return {"id": sample_id, "score": 1.0 if ok else 0.0, "oos": True,
                "gate": ok, "f1": 1.0 if ok else 0.0, "n_sub": len(sub_calls), "n_gold": 0}

    gold_env = env_factory(sample_id)
    for c in gold_calls:
        gold_env.call(c["name"], c.get("arguments") or {})
    gold_hash = gold_env.get_state_hash()

    sub_env = env_factory(sample_id)
    for c in sub_calls:
        sub_env.call(c.get("name", ""), c.get("arguments") or {})
    sub_hash = sub_env.get_state_hash()

    gate = sub_hash == gold_hash
    m = max_matching(sub_calls, gold_calls, fuzzy) if sub_calls else 0
    p = m / len(sub_calls) if sub_calls else 0.0
    r = m / len(gold_calls)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"id": sample_id, "score": f1 if gate else 0.0, "oos": False, "gate": gate,
            "f1": f1, "matched": m, "n_sub": len(sub_calls), "n_gold": len(gold_calls)}


def run(split: str, root: Path, submission: Path | None, self_check: bool, limit: int | None, dev: Path | None = None):
    sys.path.insert(0, str(root / split))
    from zaic_tools import ToolEnv  # noqa: E402  — bản zaic_tools CỦA SPLIT, không hard-code

    fuzzy = load_fuzzy_params(root / split / f"{split}_tools_catalog.json")
    gold_by_id, order = {}, []
    with open(root / split / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            if "label" not in it:
                sys.exit(f"{split}.jsonl không có nhãn — chỉ chấm được trên split có label (train).")
            gold_by_id[it["id"]] = it["label"]["tool_calls"]
            order.append(it["id"])
    if dev:
        dev_ids = set(json.loads(dev.read_text(encoding="utf-8"))["dev"])
        order = [i for i in order if i in dev_ids]
    if limit:
        order = order[:limit]

    if self_check:
        subs = {i: gold_by_id[i] for i in order}
    else:
        subs = {}
        with open(submission, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    subs[row["id"]] = row.get("tool_calls") or []

    rows = [score_sample(lambda sid: ToolEnv(split, sid, root=root), sid,
                         subs.get(sid, []), gold_by_id[sid], fuzzy) for sid in order]

    n = len(rows)
    oos = [r for r in rows if r["oos"]]
    reg = [r for r in rows if not r["oos"]]
    gated = [r for r in reg if r["gate"]]
    print(f"\n{'='*62}\nSPLIT {split} — {n} mẫu" + ("  [SELF-CHECK: gold chấm gold]" if self_check else ""))
    print("=" * 62)
    print(f"ĐIỂM TỔNG                 {sum(r['score'] for r in rows)/n:.4f}")
    print(f"  out-of-scope  {len(oos):4d} mẫu   đúng {sum(r['gate'] for r in oos):4d}"
          f"  ({sum(r['score'] for r in oos)/max(len(oos),1):.3f})")
    print(f"  thường        {len(reg):4d} mẫu   qua gate {len(gated):4d}"
          f"  ({len(gated)/max(len(reg),1)*100:.1f}%)")
    print(f"  F1 khi đã qua gate        {sum(r['f1'] for r in gated)/max(len(gated),1):.4f}")
    print(f"  F1 bỏ qua gate (tham khảo){sum(r['f1'] for r in reg)/max(len(reg),1):.4f}")
    if self_check:
        bad = [r for r in rows if r["score"] < 1.0]
        print(f"\n{'✓ ĐẠT — bộ chấm khớp gold hoàn toàn' if not bad else f'✗ LỖI — {len(bad)} mẫu không đạt 1.0'}")
        for r in bad[:10]:
            print("   ", r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--sub", type=Path, help="submission.jsonl cần chấm")
    ap.add_argument("--self-check", action="store_true", help="chấm gold bằng gold, phải ra 1.0")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dev", type=Path, help="split_dev.json — chỉ chấm dev set")
    a = ap.parse_args()
    if not a.sub and not a.self_check:
        ap.error("cần --sub hoặc --self-check")
    run(a.split, a.root, a.sub, a.self_check, a.limit, a.dev)
