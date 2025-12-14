#!/usr/bin/env python3
# news_parser_merge.py
"""
Экспорт сообщений из телеграм-канала с фильтрацией и склеиванием рекурсивных ссылок (reply chains).
Промо/рекламу НЕ удаляем целиком по умолчанию — они вырезаются из текста и сохраняются в removed_promos.
При merge_replies склейка записывается в поле "text", предки не сохраняются как отдельные записи.
Особенность: если вы указываете `--limit N` (N > 0), скрипт теперь будет пытаться
вернуть ровно N итоговых записей (если доступно) — он будет динамически загружать
больше сообщений, пока не достигнет результата или пока не исчерпает историю.
"""

import argparse
import asyncio
import json
import os
import re
from typing import Dict, Tuple, Optional, Any, Set, List

from telethon import TelegramClient
from telethon.errors import RPCError
from telethon.tl.types import Message, PeerChannel, PeerUser


# ---------------------------
# Настройки промо-шаблонов (можно дополнять)
PROMO_PATTERNS = [
    # Подписки и призывы
    r"Подпис(аться|ывайтесь|ка|ки|ы)\b.*",
    r"Не пропустите\b.*",
    r"Читайте нас\b.*",
    r"Читать нас\b.*",
    r"Присоединяйтесь!?",
    r"Смотрите прямо сейчас\b.*",
    r"\s*Подписывайтесь\b.*",
    r"Чтобы не пропустить\b.*",
    r"Чтобы не пропустить наши эксклюзивы\b.*",
    # Упоминания каналов и платформ
    r"видео в канале\b.*",
    r"в канале Председателя ГД в MAX:?",
    r"видео в канале Председателя ГД в MAX:?",
    r"Подробнее\b.*",
    r"Подробнее\s*(?:–|-|:)?\s*(?:на сайте Правительства)?\.?",
    # Ссылки (любые)
    r"https?://\S+",
    r"\b\w+\.(ru|com|org|net|cc|me|io)\S*",
]
PROMO_RE = [re.compile(p, re.I) for p in PROMO_PATTERNS]
# ---------------------------


def parse_args():
    """
    Парсит аргументы командной строки.
    """
    p = argparse.ArgumentParser(
        description="Export Telegram channel messages to JSON with filtering + merge replies."
    )
    p.add_argument(
        "--api-id",
        type=int,
        help="Telegram API ID (required unless session already has auth).",
    )
    p.add_argument(
        "--api-hash", help="Telegram API HASH (required unless session already has auth)."
    )
    p.add_argument(
        "--session", default="tg_export_session", help="Telethon session name (file)."
    )
    p.add_argument(
        "--channel",
        required=True,
        help="Channel username, invite link, or ID (e.g. @channelname or https://t.me/channelname).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Target final number of processed messages (default 100). If 0 -> no target, process whatever is fetched.",
    )
    p.add_argument("--out", default="messages.json", help="Output JSON filename.")
    p.add_argument(
        "--download-media", action="store_true", help="Also download message media to disk."
    )
    p.add_argument(
        "--media-dir",
        default="media",
        help="Directory to save media files (if --download-media).",
    )
    # Filtering options
    p.add_argument(
        "--min-text-len",
        type=int,
        default=30,
        help="Минимальная длина текста, иначе сообщение считается шумом (default 10).",
    )
    p.add_argument(
        "--drop-media-only",
        action="store_true",
        help="Удалять сообщения с media и пустым текстом.",
    )
    p.add_argument(
        "--drop-promos",
        action="store_true",
        help="Если установлен, сообщения с совпадением промо-шаблонов будут полностью удалены. По умолчанию промо вырезаются.",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Удалять дубликаты по тексту+media_type (простая дедупликация).",
    )
    # Merge / reply options
    p.add_argument(
        "--merge-replies",
        action="store_true",
        help="Склеивать сообщения с их предками по цепочке reply_to (склейка записывается в поле 'text').",
    )
    p.add_argument(
        "--max-merge-depth",
        type=int,
        default=10,
        help="Максимальная глубина рекурсивного склейвания reply-цепочки (default 10).",
    )
    p.add_argument(
        "--fetch-referenced",
        action="store_true",
        help="При склейке подтягивать отсутствующие в основной выборке ссылочные сообщения по API.",
    )
    p.add_argument(
        "--include-referenced",
        action="store_true",
        help="Если referenced сообщение было отброшено фильтром — подтянуть его для склейки (но НЕ сохранять отдельно).",
    )
    return p.parse_args()


def message_to_dict(msg: Message, entity=None) -> dict:
    """
    Преобразует Telethon Message -> сериализуемый словарь.
    """
    reply_to = None
    try:
        reply_to = getattr(msg, "reply_to_msg_id", None)
    except Exception:
        reply_to = None

    fwd_info = {}
    try:
        f = getattr(msg, "fwd_from", None)
        if f:
            # from_id может быть PeerChannel / PeerUser
            if isinstance(f.from_id, PeerChannel):
                fwd_info["from_channel_id"] = f.from_id.channel_id
            elif isinstance(f.from_id, PeerUser):
                fwd_info["from_user_id"] = f.from_id.user_id
            else:
                fwd_info["from_id"] = None
            # channel_id — обычно int, но проверим
            fwd_info["channel_id"] = int(f.channel_id) if f.channel_id else None
            fwd_info["channel_post"] = int(f.channel_post) if f.channel_post else None
            fwd_info["hidden"] = bool(f.hidden)
    except Exception:
        fwd_info = {}

    text = getattr(msg, "message", "") or ""
    data = {
        "id": getattr(msg, "id", None),
        "date": msg.date.isoformat() if getattr(msg, "date", None) else None,
        "text": text,
        "raw_text_len": len(text),
        "views": getattr(msg, "views", None),
        "forwards": getattr(msg, "forwards", None),
        "reply_to_msg_id": reply_to,
        "is_reply": bool(reply_to),
        "sender_id": getattr(msg, "sender_id", None),
        "has_media": bool(getattr(msg, "media", None)),
        "fwd_info": fwd_info,
    }
    if entity is not None:
        data["channel_id"] = getattr(entity, "id", None)
        data["channel_title"] = getattr(entity, "title", None)
        data["channel_username"] = getattr(entity, "username", None)
    return data


def strip_promotional_text(text: str) -> Tuple[str, List[str]]:
    """
    Убирает из текста все вхождения шаблонов PROMO_RE.
    Возвращает (cleaned_text, removed_fragments_list).
    """
    if not text:
        return text, []

    removed: List[str] = []
    for rx in PROMO_RE:
        for m in rx.finditer(text):
            s = m.group(0)
            if s and s.strip():
                removed.append(s)

    cleaned = text
    for rx in PROMO_RE:
        cleaned = rx.sub("", cleaned)

    EMOJI_RE = re.compile(
        "["
        u"\U0001F300-\U0001F5FF"
        u"\U0001F600-\U0001F64F"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U00002000-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )
    cleaned = EMOJI_RE.sub("", cleaned)
    cleaned = "\n".join([line.strip() for line in cleaned.splitlines()])
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    uniq_removed = []
    seen = set()
    for s in removed:
        key = s.strip().lower()
        if key and key not in seen:
            uniq_removed.append(s.strip())
            seen.add(key)

    return cleaned, uniq_removed


def is_noise(item: dict, min_text_len: int, drop_media_only: bool) -> Tuple[bool, str]:
    """
    Эвристика: возвращает (True, reason) если сообщение шум.
    """
    text = (item.get("text") or "").strip()
    length = item.get("raw_text_len", len(text))

    if length == 0:
        if drop_media_only and item.get("has_media"):
            return True, "empty_text_and_media (drop_media_only)"
        return True, "empty_text"

    if length < min_text_len:
        return True, f"too_short ({length} < {min_text_len})"

    alnum = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", text)
    if len(alnum) < 10 and length < (min_text_len * 2):
        return True, "mostly_non_alnum"

    if re.fullmatch(r"(https?://\S+|\S+\.\S+)", text.strip()):
        return True, "url_only"

    return False, ""


async def fetch_messages(
    client: TelegramClient,
    entity,
    limit: Optional[int],
    download_media: bool,
    media_dir: str,
) -> Dict[int, Tuple[dict, Message]]:
    """
    Загружает сообщения и возвращает mapping id -> (serializable_dict, raw_message_obj).
    Если limit is None -> fetch all (careful).
    """
    mapping: Dict[int, Tuple[dict, Message]] = {}
    fetch_limit = None if (limit is None or limit == 0) else limit

    if download_media:
        os.makedirs(media_dir, exist_ok=True)

    async for msg in client.iter_messages(entity, limit=fetch_limit):
        try:
            item = message_to_dict(msg, entity=entity)
            mapping[item["id"]] = (item, msg)
        except Exception as e:
            print(f"Warning: failed to process msg id={getattr(msg, 'id', None)}: {e}")

    return mapping


async def fetch_single_message_by_id(
    client: TelegramClient, entity, msg_id: int
) -> Optional[Tuple[dict, Message]]:
    """
    Попытка получить сообщение по id у заданного entity. Возвращает (item, msg) или None.
    """
    try:
        fetched = await client.get_messages(entity, ids=msg_id)
        if fetched:
            item = message_to_dict(fetched, entity=entity)
            return item, fetched
    except Exception as e:
        # не шумим сильно, просто логим
        print(f"Info: couldn't fetch referenced msg id={msg_id}: {e}")
    return None


async def merge_reply_chain_for_message(
    msg_id: int,
    mapping: Dict[int, Tuple[dict, Message]],
    client: TelegramClient,
    entity,
    max_depth: int,
    fetch_referenced: bool,
    include_referenced: bool,
    seen: Optional[Set[int]] = None,
) -> Tuple[str, list]:
    """
    Рекурсивно собирает цепочку предков (reply_to) и формирует merged text (root -> ... -> leaf).
    Возвращает (merged_text, chain_ids_root_to_leaf).
    """
    if seen is None:
        seen = set()

    parts = []
    chain_ids = []
    current_id = msg_id
    depth = 0

    while current_id is not None and depth < max_depth:
        if current_id in seen:
            break
        seen.add(current_id)

        if current_id in mapping:
            item, _ = mapping[current_id]
            parts.append((current_id, item.get("text", "")))
            chain_ids.append(current_id)
            current_id = item.get("reply_to_msg_id")
            depth += 1
            continue

        if fetch_referenced:
            fetched = await fetch_single_message_by_id(client, entity, current_id)
            if fetched:
                item, raw_msg = fetched
                mapping[current_id] = (item, raw_msg)
                parts.append((current_id, item.get("text", "")))
                chain_ids.append(current_id)
                current_id = item.get("reply_to_msg_id")
                depth += 1
                continue

        break

    parts.reverse()
    chain_ids.reverse()

    merged_blocks = []
    for pid, txt in parts:
        merged_blocks.append(txt or "")

    merged_text = "\n\n".join(x for x in merged_blocks if x).strip()
    return merged_text, chain_ids


def collect_all_referenced_ids(
    mapping: Dict[int, Tuple[dict, Message]]
) -> Set[int]:
    """
    Собирает множество id, на которые кто-либо ссылается (reply_to_msg_id).
    Эти сообщения считаются ПРЕДКАМИ и НЕ будут сохранены отдельно (мы предпочитаем листы).
    """
    referenced = set()
    for item, _ in mapping.values():
        rid = item.get("reply_to_msg_id")
        if isinstance(rid, int) and rid is not None:
            referenced.add(rid)
    return referenced


def process_raw_mapping(
    raw_mapping: Dict[int, Tuple[dict, Message]],
    args,
    client,
    entity,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Выполняет очистку промо, фильтрацию, дедупликацию, удаление предков (оставляем листы),
    и (если merge_replies) помечает места для последующей склейки (но не делает fetch здесь).
    Возвращает (output_messages_list, stats).
    NOTE: merge-replies actual merging (fetching referenced) is performed later in async flow
    because it may require await calls. Здесь мы только подготавливаем набор final_ids и базовые поля.
    """
    # 2) Вырезаем промо (сохраняем оригинал и removed_promos)
    for mid, (item, raw_msg) in raw_mapping.items():
        original = item.get("text", "") or ""
        item["_original_text"] = original
        item["removed_promos"] = []
        if original:
            cleaned, removed = strip_promotional_text(original)
            item["text"] = cleaned
            item["raw_text_len"] = len(cleaned)
            item["removed_promos"] = removed

        # --- НОВОЕ: если в оригинале есть хештег #реклама — помечаем чтобы удалить сообщение полностью
        # регистронезависимая проверка
        try:
            if re.search(r"(?i)(?:^|\s)#реклама\b", original):
                item["_has_hashtag_реклама"] = True
            else:
                item["_has_hashtag_реклама"] = False
        except Exception:
            item["_has_hashtag_реклама"] = False

    # ------------------------------------------------------------
    # 3) Фильтрация (первичный)
    kept_ids = set()
    dropped_ids = set()
    drop_reasons = {}
    for mid, (item, raw_msg) in raw_mapping.items():
        # Сначала — жёсткое правило: если найден '#реклама' в тексте — удаляем полностью
        if item.get("_has_hashtag_реклама"):
            dropped_ids.add(mid)
            drop_reasons[mid] = "contains_hashtag_ad (#реклама)"
            continue

        if args.drop_promos:
            orig = item.get("_original_text", "") or ""
            promo_found = any(rx.search(orig) for rx in PROMO_RE)
            if promo_found:
                dropped_ids.add(mid)
                drop_reasons[mid] = "promo_whole_message (drop_promos)"
                continue

        noisy, reason = is_noise(item, args.min_text_len, args.drop_media_only)
        if noisy:
            dropped_ids.add(mid)
            drop_reasons[mid] = reason
        else:
            kept_ids.add(mid)

    # 4) Дедупликация простая (опционально)
    final_ids: List[int] = []
    if args.dedupe:
        seen_hashes = set()
        for mid in sorted(kept_ids):
            item = raw_mapping[mid][0]
            key = (item.get("text", ""), "media" if item.get("has_media") else "nomedia")
            if key in seen_hashes:
                dropped_ids.add(mid)
                drop_reasons[mid] = "duplicate"
            else:
                seen_hashes.add(key)
                final_ids.append(mid)
    else:
        final_ids = sorted(kept_ids)

    # 4.5) Удаляем ПРЕДКОВ (оставляем только ЛИСТЫ ответов)
    referenced_ids = collect_all_referenced_ids(raw_mapping)
    final_ids = [mid for mid in final_ids if mid not in referenced_ids]

    # Формируем базовый output (без merge-replies склейки, которая требует await)
    output_messages = []
    for mid in final_ids:
        item, raw_msg = raw_mapping.get(mid, (None, None))
        if not item:
            continue
        out_item = dict(item)  # shallow copy
        if mid in drop_reasons:
            out_item["_was_dropped_reason"] = drop_reasons[mid]
        output_messages.append(out_item)

    stats = {
        "fetched_raw": len(raw_mapping),
        "initial_kept": len(kept_ids),
        "initial_dropped": len(dropped_ids),
        "final_candidates": len(final_ids),
    }
    return output_messages, stats


async def main():
    args = parse_args()

    if (
        args.api_id is None or args.api_hash is None
    ) and not os.path.exists(args.session + ".session"):
        raise SystemExit("API ID и API HASH требуются, если нет сохранённой session.")

    client = TelegramClient(args.session, args.api_id, args.api_hash)
    try:
        await client.start()
    except Exception as e:
        raise SystemExit(f"Failed to start Telegram client: {e}")

    try:
        entity = await client.get_entity(args.channel)
        print("Channel resolved:")
        print(f" id: {getattr(entity, 'id', None)}")
        print(f" title: {getattr(entity, 'title', None)!s}")
        print(f" username: {getattr(entity, 'username', None)!s}")

        target = args.limit if args.limit and args.limit > 0 else None
        # If target is set, we'll progressively fetch larger batches until we can produce at least target final messages
        # (or until history is exhausted or max_fetch reached). If no target, fetch once with args.limit (as before).
        max_fetch_cap = 5000  # safety cap to avoid huge scans; adjust if you need deeper history
        fetch_size = max(max(200, (target or 100) * 2), 200)

        last_raw_mapping = {}
        stats_accum = None
        output_messages = []

        if target is None:
            # Classical single fetch (respecting args.limit as fetch limit)
            raw_mapping = await fetch_messages(
                client, entity, args.limit, args.download_media, args.media_dir
            )
            output_messages, stats = process_raw_mapping(
                raw_mapping, args, client, entity
            )
            # If merge-replies requires fetching referenced messages, perform it here for each output message
            if args.merge_replies:
                # Create raw_mapping local reference (may be used by merge to fetch referenced)
                for out in output_messages:
                    mid = out["id"]
                    merged_text, chain_ids = await merge_reply_chain_for_message(
                        mid,
                        raw_mapping,
                        client,
                        entity,
                        max_depth=args.max_merge_depth,
                        fetch_referenced=args.fetch_referenced,
                        include_referenced=args.include_referenced,
                    )
                    if merged_text:
                        out["text"] = merged_text
                        out["raw_text_len"] = len(merged_text)

            stats["final_saved"] = len(output_messages)
            stats_accum = stats
            last_raw_mapping = raw_mapping

        else:
            # Progressive fetching: increase fetch_size until we have >= target final outputs or exhaust history
            exhausted = False
            while True:
                if fetch_size > max_fetch_cap:
                    fetch_size = max_fetch_cap

                print(
                    f"Fetching batch of {fetch_size} messages to try reach target {target} ..."
                )
                raw_mapping = await fetch_messages(
                    client, entity, fetch_size, args.download_media, args.media_dir
                )
                print(f"Fetched {len(raw_mapping)} raw messages (batch).")

                output_candidates, stats = process_raw_mapping(
                    raw_mapping, args, client, entity
                )
                # If merge required, we need to run merge for each candidate to ensure parent content is included.
                if args.merge_replies:
                    # for merging we need the live mapping (for fetch_referenced) — use raw_mapping
                    for out in output_candidates:
                        mid = out["id"]
                        merged_text, chain_ids = await merge_reply_chain_for_message(
                            mid,
                            raw_mapping,
                            client,
                            entity,
                            max_depth=args.max_merge_depth,
                            fetch_referenced=args.fetch_referenced,
                            include_referenced=args.include_referenced,
                        )
                        if merged_text:
                            out["text"] = merged_text
                            out["raw_text_len"] = len(merged_text)

                # Sort candidates by date desc (most recent first)
                def parse_date_iso(it):
                    try:
                        return it.get("date") or ""
                    except Exception:
                        return ""

                output_candidates_sorted = sorted(
                    output_candidates, key=lambda x: x.get("date", ""), reverse=True
                )

                if len(output_candidates_sorted) >= target:
                    # take top 'target' most recent
                    output_messages = output_candidates_sorted[:target]
                    stats["final_saved"] = len(output_messages)
                    stats_accum = stats
                    last_raw_mapping = raw_mapping
                    break

                # Not enough yet: see if we exhausted history (fetched less than fetch_size)
                if len(raw_mapping) < fetch_size:
                    # exhausted history — take whatever we have
                    output_messages = output_candidates_sorted
                    stats["final_saved"] = len(output_messages)
                    stats_accum = stats
                    last_raw_mapping = raw_mapping
                    exhausted = True
                    break

                # otherwise increase fetch_size and retry
                # cap to max_fetch_cap to avoid runaway
                if fetch_size >= max_fetch_cap:
                    output_messages = output_candidates_sorted
                    stats["final_saved"] = len(output_messages)
                    stats_accum = stats
                    last_raw_mapping = raw_mapping
                    break

                fetch_size = min(fetch_size * 2, max_fetch_cap)
                # loop repeats

        # Final safety: ensure unique messages by id and at most target if specified
        seen_ids = set()
        final_output_unique = []
        for out in output_messages:
            if out["id"] in seen_ids:
                continue
            seen_ids.add(out["id"])
            final_output_unique.append(out)
            if target and len(final_output_unique) >= target:
                break

        # If we still have fewer than target and target set, attempt to fill from other candidates:
        if target and len(final_output_unique) < target:
            # Try to include kept-but-previously-removed parents (as last resort), preserving no duplicates.
            # We'll add parent items (raw_mapping entries) that are not yet included, up to target.
            for mid, (item, raw_msg) in sorted(last_raw_mapping.items(), reverse=True):
                if mid in seen_ids:
                    continue
                # skip those that were explicitly dropped (we prefer kept ones)
                # include them only if they pass basic noise test
                noisy, _ = is_noise(item, args.min_text_len, args.drop_media_only)
                if noisy:
                    continue
                final_output_unique.append(dict(item))
                seen_ids.add(mid)
                if len(final_output_unique) >= target:
                    break

        # Final sort: make consistent ordering by date descending
        final_output_unique = sorted(
            final_output_unique, key=lambda x: x.get("date", ""), reverse=True
        )

        # Write to file
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        data_out = os.path.join(data_dir, os.path.basename(args.out))
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(final_output_unique, f, ensure_ascii=False, indent=2)

        # Print stats
        if stats_accum is None:
            stats_accum = {
                "fetched_raw": len(last_raw_mapping),
                "initial_kept": 0,
                "initial_dropped": 0,
                "final_saved": len(final_output_unique),
            }
        else:
            stats_accum["final_saved"] = len(final_output_unique)

        print(f"Saved {len(final_output_unique)} messages to {data_out}")
        print(
            f"Stats: fetched_raw={stats_accum.get('fetched_raw')}, "
            f"initial_kept={stats_accum.get('initial_kept')}, "
            f"initial_dropped={stats_accum.get('initial_dropped')}, "
            f"final_saved={stats_accum.get('final_saved')}"
        )
        if args.merge_replies:
            print(
                "Note: merge_replies applied and merged content is stored in 'text' field for each saved message."
            )

    except RPCError as e:
        raise SystemExit(f"Telegram RPC error: {e}")
    except Exception as e:
        raise SystemExit(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())