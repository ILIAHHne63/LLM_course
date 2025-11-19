#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path


def format_block(title: str, body: str) -> str:
    if not body:
        return f"{title}: (нет данных)"
    wrapped = textwrap.fill(body.strip(), width=100)
    return f"{title}:\n{wrapped}\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretty-print summary information from the latest saved answer"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to a specific JSON file. If omitted, the most recent file in outputs/ is used.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1] / "outputs"
    if args.path:
        path = Path(args.path)
    else:
        candidates = sorted(base_dir.glob("answer_*.json"), reverse=True)
        if not candidates:
            print("[show_summary] No answer files found in outputs/.", file=sys.stderr)
            return 1
        path = candidates[0]
    if not path.exists():
        print(f"[show_summary] File '{path}' not found.", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 80)
    print(f"Запрос: {data.get('query')}")
    print("=" * 80)
    print(format_block("Сводка", data.get("summary") or ""))

    extracted = data.get("extracted_information")
    if extracted:
        print(format_block("Извлечённые факты", extracted))

    results = data.get("results", [])
    if not results:
        print("Нет новостных сообщений в ответе.")
        return 0

    print("Новости:")
    for idx, item in enumerate(results, 1):
        channel = item.get("channel_title") or item.get("channel_username") or "Неизвестный источник"
        date = item.get("date") or "Нет даты"
        if item.get('score'): header = f"{idx}. {date} — {channel} (score={item.get('score')})"
        else: header = f"{idx}. {date} — {channel}"
        print(header)
        text = (item.get("text") or "").replace("\n", " ").strip()
        wrapped = textwrap.fill(text, width=100, initial_indent="   ", subsequent_indent="   ")
        print(wrapped)
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
