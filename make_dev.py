"""Tách train thành train/dev theo APP, mô phỏng tình huống "app chưa từng thấy".

public_test đưa vào 11 app hoàn toàn mới mà train không có (ecommerce, parking,
stocks, grocery...). private_test gần như chắc chắn cũng vậy. Để đo được khả năng
tổng quát hoá TRƯỚC khi nộp, ta giữ lại 10 app ngoại vi của train làm dev: mọi mẫu
mà gold chạm tới một trong các app đó -> dev, phần còn lại -> train.

Chọn app held-out theo đúng tính chất của 11 app mới ở public_test:
  - "tìm rồi đặt/mua": ride_hailing, food_delivery, flights, bike_rental
  - "danh sách / theo dõi": shopping_list, library, loyalty, fitness
  - "tra cứu tiện ích":   maps, package_tracking
KHÔNG giữ lại các app lõi (contacts, sys, calendar, smart_home...) vì public_test
vẫn dùng chúng — chỉ các app ngoại vi mới bị thay.

    python make_dev.py          # ghi split_dev.json
"""
from __future__ import annotations

import json
from pathlib import Path

HELD_OUT = ["ride_hailing", "food_delivery", "flights", "bike_rental",
            "shopping_list", "library", "loyalty", "fitness",
            "maps", "package_tracking"]

OUT = Path("split_dev.json")


def main(split: str = "train", root: Path = Path(".")):
    rows = [json.loads(l) for l in open(root / split / f"{split}.jsonl", encoding="utf-8") if l.strip()]
    held = set(HELD_OUT)

    dev, tr = [], []
    for r in rows:
        apps = {c["name"].split(".", 1)[0] for c in r["label"]["tool_calls"]}
        (dev if apps & held else tr).append(r["id"])

    by_id = {r["id"]: r for r in rows}

    def stats(ids):
        n_oos = sum(1 for i in ids if not by_id[i]["label"]["tool_calls"])
        n_w = sum(1 for i in ids if by_id[i]["label"]["tool_calls"])
        return n_oos, n_w

    print(f"giữ lại {len(held)} app: {', '.join(sorted(held))}\n")
    print(f"  train  {len(tr):5d} mẫu  ({len(tr)/len(rows)*100:.1f}%)   out-of-scope {stats(tr)[0]}")
    print(f"  dev    {len(dev):5d} mẫu  ({len(dev)/len(rows)*100:.1f}%)   out-of-scope {stats(dev)[0]}")

    # Mẫu out-of-scope không chạm app nào nên luôn rơi vào train — tách riêng một phần sang dev
    # để dev vẫn đo được năng lực nhận biết ngoài-phạm-vi.
    oos = [i for i in tr if not by_id[i]["label"]["tool_calls"]]
    move = oos[::3]
    tr = [i for i in tr if i not in set(move)]
    dev += move
    print(f"\n  chuyển {len(move)} mẫu out-of-scope sang dev để dev đo được cả lớp này")
    print(f"  => train {len(tr)} | dev {len(dev)}")

    OUT.write_text(json.dumps({"held_out_apps": sorted(held), "train": tr, "dev": dev},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nđã ghi {OUT}")


if __name__ == "__main__":
    main()
