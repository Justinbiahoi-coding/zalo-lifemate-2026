#!/usr/bin/env python3
"""Augment SFT data with schema anonymization.

For each training sample, creates anonymized copies where app/tool names
are randomly replaced. This forces the model to READ the schema instead
of memorizing "smart_home.light_set needs device_id".

    python augment_sft.py --input sft_train.jsonl --output sft_aug.jsonl --copies 2
"""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path

_FAKE_APPS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
    "phoenix", "atlas", "cosmos", "nebula", "vertex", "prism", "nexus",
    "orbit", "spark", "flux", "pulse", "drift", "crest", "forge",
    "grove", "haven", "iris", "jade", "lumen", "maple", "nova",
]


_TYPE_NAMES = {"str", "int", "num", "bool"}


def _extract_apps(sys_content: str) -> list[str]:
    return [a for a in re.findall(r'^\[(\w+)\]', sys_content, re.MULTILINE)
            if a not in _TYPE_NAMES]


def _build_map(apps: list[str], rng: random.Random) -> dict[str, str]:
    pool = list(_FAKE_APPS)
    rng.shuffle(pool)
    return {app: pool[i % len(pool)] for i, app in enumerate(apps)}


def anonymize(messages: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    sys_content = messages[0]["content"]
    apps = _extract_apps(sys_content)
    if not apps:
        return messages

    mapping = _build_map(apps, rng)

    def _replace(text: str) -> str:
        for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
            text = text.replace(f"[{old}]", f"[{new}]")
            text = text.replace(f"{old}.", f"{new}.")
        return text

    out = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "system":
            content = _replace(content)
        elif msg["role"] == "assistant" and "CALL:" in content:
            content = _replace(content)
        elif msg["role"] == "user" and "→" in content:
            content = _replace(content)
        out.append({"role": msg["role"], "content": content})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("sft_train.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("sft_aug.jsonl"))
    ap.add_argument("--copies", type=int, default=2,
                    help="number of anonymized copies per sample (0=originals only)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    print(f"Input: {len(rows)} samples")

    rng = random.Random(a.seed)
    out = []
    for row in rows:
        out.append(row)
        msgs = row["messages"]
        has_calls = any("CALL:" in m.get("content", "") for m in msgs)
        if has_calls and a.copies > 0:
            for c in range(a.copies):
                seed = rng.randint(0, 2**31)
                anon_msgs = anonymize(msgs, seed)
                out.append({"messages": anon_msgs})

    rng2 = random.Random(a.seed + 1)
    rng2.shuffle(out)

    with open(a.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_orig = len(rows)
    n_aug = len(out) - n_orig
    print(f"Output: {len(out)} samples ({n_orig} original + {n_aug} anonymized)")
    print(f"→ {a.output}")


if __name__ == "__main__":
    main()
