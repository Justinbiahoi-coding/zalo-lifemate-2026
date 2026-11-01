"""Vietnamese number/datetime normalizer for LifeMate.

Appends [chuẩn hoá: ...] hints to audio_text so the model sees
both original text and normalized values.
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta

_DIGITS = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8,
    "chín": 9, "mốt": 1,
}
_TENS = {"mười": 10, "mươi": 10}
_MULT = {"trăm": 100, "nghìn": 1000, "ngàn": 1000, "triệu": 1_000_000, "tỷ": 1_000_000_000}
_DOW = {"hai": 1, "ba": 2, "tư": 3, "năm": 4, "sáu": 5, "bảy": 6}


def _tokenize(text: str):
    tokens = []
    for w in text.lower().split():
        if w.isdigit():
            tokens.append(("n", int(w)))
        elif w in _DIGITS:
            tokens.append(("n", _DIGITS[w]))
        elif w in _TENS:
            tokens.append(("t",))
        elif w in _MULT:
            tokens.append(("m", _MULT[w]))
        elif w in ("linh", "lẻ"):
            pass
        elif w == "rưỡi":
            tokens.append(("half",))
        else:
            return None
    return tokens


def _parse_small(tokens):
    """Parse number < 1000 from tokens (handles trăm/mươi/mười)."""
    result = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok[0] == "n":
            if i + 1 < len(tokens) and tokens[i + 1][0] == "m" and tokens[i + 1][1] == 100:
                result += tok[1] * 100
                i += 2
            elif i + 1 < len(tokens) and tokens[i + 1][0] == "t":
                result += tok[1] * 10
                i += 2
            else:
                result += tok[1]
                i += 1
        elif tok[0] == "t":
            result += 10
            i += 1
        elif tok[0] == "half":
            i += 1
        else:
            i += 1
    return result


def parse_vn_number(text: str) -> int | None:
    tokens = _tokenize(text)
    if not tokens:
        return None

    groups = []
    sub = []
    last_major = None

    for tok in tokens:
        if tok[0] == "m" and tok[1] >= 1000:
            val = _parse_small(sub) if sub else 1
            groups.append((val, tok[1]))
            last_major = tok[1]
            sub = []
        else:
            sub.append(tok)

    tail_val = _parse_small(sub) if sub else 0

    has_half = any(t[0] == "half" for t in sub)

    result = sum(v * m for v, m in groups)

    if tail_val > 0 and groups:
        prev_mult = groups[-1][1]
        if tail_val < 10 and prev_mult >= 1000:
            result += tail_val * (prev_mult // 10)
        else:
            result += tail_val
    elif tail_val > 0:
        result += tail_val

    if has_half:
        if last_major:
            result += last_major // 2
        elif result > 0:
            result = int(result * 1.5)

    return result if result > 0 else None


def _digit(s: str) -> int:
    if s.isdigit():
        return int(s)
    return _DIGITS.get(s.lower(), 0)


def _to_24h(h: int, period: str | None) -> int:
    if not period:
        return h
    p = period.lower()
    if p in ("chiều", "tối") and h < 12:
        return h + 12
    if p == "đêm":
        return 0 if h == 12 else h
    if p == "trưa":
        return 12 if h == 12 else h
    return h


_DIGIT_ALT = '|'.join(sorted(_DIGITS, key=len, reverse=True))
_PERIOD_ALT = 'sáng|trưa|chiều|tối|đêm'

_RUOI_RE = re.compile(
    rf'(\d+|{_DIGIT_ALT})\s*(?:giờ\s*)?rưỡi(?:\s*({_PERIOD_ALT}))?',
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    rf'(\d+|{_DIGIT_ALT})\s*giờ'
    rf'(?:\s*(?:kém\s*)?(\d+|{_DIGIT_ALT})(?:\s*phút)?)?'
    rf'(?:\s*({_PERIOD_ALT}))?',
    re.IGNORECASE,
)


def parse_time(text: str) -> str | None:
    m = _RUOI_RE.search(text)
    if m:
        h = _digit(m.group(1))
        h = _to_24h(h, m.group(2))
        return f"{h:02d}:30"

    m = _TIME_RE.search(text)
    if not m:
        return None
    h = _digit(m.group(1))
    mins = 0
    if m.group(2):
        mins = _digit(m.group(2))
    if "kém" in m.group(0):
        h -= 1
        mins = 60 - mins
    h = _to_24h(h, m.group(3))
    return f"{h:02d}:{mins:02d}"


def parse_relative_date(text: str, now: datetime) -> str | None:
    t = text.lower()
    today = now.date()
    if re.search(r'\bngày\s*kia\b|\bngày\s*mốt\b', t):
        return (today + timedelta(days=2)).isoformat()
    if re.search(r'\bmốt\b', t) and not re.search(r'\bmốt\b.*(?:giờ|rưỡi|phút)', t):
        return (today + timedelta(days=2)).isoformat()
    if re.search(r'\bmai\b', t):
        return (today + timedelta(days=1)).isoformat()
    if re.search(r'\bhôm\s*nay\b', t):
        return today.isoformat()
    m = re.search(r'\bthứ\s*(hai|ba|tư|năm|sáu|bảy)\b', t)
    if m:
        target = _DOW[m.group(1)]
        delta = target - today.isoweekday()
        if "tuần sau" in t or "tuần tới" in t:
            delta += 7
        if delta <= 0:
            delta += 7
        return (today + timedelta(days=delta)).isoformat()
    if re.search(r'\bchủ\s*nhật\b', t):
        delta = 7 - today.isoweekday()
        if "tuần sau" in t or "tuần tới" in t:
            delta += 7
        if delta <= 0:
            delta += 7
        return (today + timedelta(days=delta)).isoformat()
    return None


def parse_duration(text: str) -> int | None:
    t = text.lower()
    if re.search(r'\bnửa\s*tiếng\b', t):
        return 30
    m = re.search(rf'(\d+|{_DIGIT_ALT})\s*tiếng(?:\s*rưỡi)?', t, re.IGNORECASE)
    if m:
        h = _digit(m.group(1))
        mins = h * 60
        if "rưỡi" in m.group(0):
            mins += 30
        return mins
    m = re.search(rf'(\d+|{_DIGIT_ALT})\s*phút', t, re.IGNORECASE)
    if m:
        return _digit(m.group(1))
    return None


_MONEY_RE = re.compile(
    rf'(\d+[\d\s]*(?:nghìn|ngàn|triệu|tỷ|trăm)(?:\s+(?:{_DIGIT_ALT}|nghìn|ngàn|triệu|tỷ|trăm))*)',
    re.IGNORECASE,
)

_WORD_NUM_PARTS = '|'.join(sorted(
    list(_DIGITS) + list(_TENS) + list(_MULT) + ["linh", "lẻ", "rưỡi"],
    key=len, reverse=True,
))
_WORD_NUM_RE = re.compile(
    rf'\b((?:{_WORD_NUM_PARTS})(?:\s+(?:{_WORD_NUM_PARTS}))+)\b',
    re.IGNORECASE,
)


def normalize(text: str, current_time: str | None = None) -> str:
    hints = []
    now = None
    if current_time:
        try:
            now = datetime.fromisoformat(current_time)
        except (ValueError, TypeError):
            pass

    t = parse_time(text)
    if t:
        hints.append(f"giờ={t}")

    if now:
        d = parse_relative_date(text, now)
        if d:
            hints.append(f"ngày={d}")

    dur = parse_duration(text)
    if dur:
        hints.append(f"thời_lượng={dur}phút")

    seen_nums = set()

    for mp in _MONEY_RE.finditer(text):
        val = parse_vn_number(mp.group(1).strip())
        if val and val >= 1000 and val not in seen_nums:
            hints.append(f"số_tiền={val}")
            seen_nums.add(val)

    for wn in _WORD_NUM_RE.finditer(text):
        matched = wn.group(1)
        if re.search(r'giờ', matched, re.IGNORECASE):
            continue
        if re.search(r'tiếng|phút', matched, re.IGNORECASE):
            continue
        if 'rưỡi' in matched.lower():
            after = text[wn.end():wn.end() + 15].strip().lower()
            if t or re.match(r'(?:sáng|trưa|chiều|tối|đêm)', after):
                continue
        val = parse_vn_number(matched)
        if val and val > 0 and val not in seen_nums:
            hints.append(f"số={val}")
            seen_nums.add(val)

    if not hints:
        return text
    return text + "\n[chuẩn hoá: " + ", ".join(hints) + "]"


if __name__ == "__main__":
    tests = [
        ("Đặt báo thức 5 rưỡi sáng mai đó", "2026-11-10T08:00:00+07:00"),
        ("chuyển hai triệu tám cho chủ nhà", None),
        ("nhắc mình 8 giờ kém 15 tối mai", "2026-11-10T08:00:00+07:00"),
        ("dời muộn nửa tiếng", None),
        ("nạp 50 nghìn điện thoại", None),
        ("chuyển một triệu hai cho Hùng", None),
        ("nhắc tám rưỡi tối nay", "2026-11-10T08:00:00+07:00"),
        ("hai trăm nghìn", None),
        ("ba mươi lăm nghìn", None),
    ]
    for text, ct in tests:
        print(f"IN:  {text}")
        print(f"OUT: {normalize(text, ct)}")
        print()
