#!/usr/bin/env python3
"""
news_parser.py
Export messages from a Telegram channel into a JSON file.

Usage examples:
  python news_parser.py --api-id 12345 --api-hash abcd... --channel @example_channel --limit 200 --out messages.json
"""

import argparse
import asyncio
import json
import os
from telethon import TelegramClient
from telethon.errors import RPCError

def parse_args():
    p = argparse.ArgumentParser(description="Export Telegram channel messages to JSON (Telethon).")
    p.add_argument("--api-id", type=int, help="Telegram API ID (required unless session already has auth).")
    p.add_argument("--api-hash", help="Telegram API HASH (required unless session already has auth).")
    p.add_argument("--session", default="tg_export_session", help="Telethon session name (file).")
    p.add_argument("--channel", required=True, help="Channel username, invite link, or ID (e.g. @channelname or https://t.me/channelname).")
    p.add_argument("--limit", type=int, default=100, help="Max number of messages to fetch (default 100). Use 0 for all available.")
    p.add_argument("--out", default="messages.json", help="Output JSON filename.")
    p.add_argument("--download-media", action="store_true", help="Also download message media to disk.")
    p.add_argument("--media-dir", default="media", help="Directory to save media files (if --download-media).")
    return p.parse_args()

def message_to_dict(msg, entity=None):
    """
    Convert a Telethon Message to a plain dict.
    If entity (channel) is provided, add channel metadata to the message dict so the
    channel title is always present per-message.
    """
    # Safe extraction of reply_to (Telethon can store reply_to_msg_id as int or object)
    reply_to = None
    try:
        # If reply_to_msg_id is an object with attribute 'reply_to_msg_id', handle it
        r = getattr(msg, "reply_to_msg_id", None)
        if r is not None:
            # If it's an object that contains attribute, try to extract numeric id
            reply_to = getattr(r, "reply_to_msg_id", r)
    except Exception:
        reply_to = None

    data = {
        "id": getattr(msg, "id", None),
        "date": msg.date.isoformat() if getattr(msg, "date", None) else None,
        "text": getattr(msg, "message", "") or "",
        "raw_text_len": len(getattr(msg, "message", "") or ""),
        "views": getattr(msg, "views", None),
        "forwards": getattr(msg, "forwards", None),
        "reply_to_msg_id": reply_to,
        "is_reply": bool(reply_to),
        "sender_id": getattr(msg, "sender_id", None),
        "has_media": bool(getattr(msg, "media", None)),
    }

    # Add channel metadata into each message to ensure channel name is always logged
    if entity is not None:
        data["channel_id"] = getattr(entity, "id", None)
        data["channel_title"] = getattr(entity, "title", None)
        data["channel_username"] = getattr(entity, "username", None)

    return data

async def fetch_messages(client, entity, limit, download_media, media_dir):
    """
    Fetch messages from a resolved entity (channel/chat) and optionally download media.
    Returns a list of message dicts.
    """
    out = []
    fetch_limit = None if (limit == 0) else limit

    if download_media:
        os.makedirs(media_dir, exist_ok=True)

    async for msg in client.iter_messages(entity, limit=fetch_limit):
        try:
            item = message_to_dict(msg, entity=entity)

            if item["has_media"]:
                media_info = {"downloaded": None, "media_type": None}
                try:
                    kind = msg.media.__class__.__name__
                    media_info["media_type"] = kind
                except Exception:
                    media_info["media_type"] = "unknown"

                if download_media:
                    base = f"{entity.id}_{msg.id}"
                    try:
                        saved = await client.download_media(msg, file=os.path.join(media_dir, base))
                        if isinstance(saved, str):
                            media_info["downloaded"] = saved
                        else:
                            fname = os.path.join(media_dir, base + ".bin")
                            with open(fname, "wb") as f:
                                f.write(saved)
                            media_info["downloaded"] = fname
                    except Exception as e:
                        media_info["download_error"] = str(e)
                item["media"] = media_info

            out.append(item)
        except Exception as e:
            out.append({"id": getattr(msg, "id", None), "error": str(e)})
    return out

async def main():
    args = parse_args()

    # Validate API credentials presence (Telethon can also re-use a session file that already has auth)
    if (args.api_id is None or args.api_hash is None) and not os.path.exists(args.session + ".session"):
        raise SystemExit("API ID and API HASH are required unless you already have a saved session file.")

    client = TelegramClient(args.session, args.api_id, args.api_hash)

    try:
        await client.start()
    except Exception as e:
        raise SystemExit(f"Failed to start Telegram client: {e}")

    try:
        # Resolve entity once and keep it (so we can always log its title)
        entity = await client.get_entity(args.channel)

        # Print channel info to stdout (immediate log)
        channel_title = getattr(entity, "title", None)
        channel_username = getattr(entity, "username", None)
        print("Channel resolved:")
        print(f"  id: {getattr(entity, 'id', None)}")
        print(f"  title: {channel_title!s}")
        print(f"  username: {channel_username!s}")

        messages = await fetch_messages(client, entity, args.limit, args.download_media, args.media_dir)
    except RPCError as e:
        raise SystemExit(f"Telegram RPC error: {e}")
    except Exception as e:
        raise SystemExit(f"Error fetching messages: {e}")
    finally:
        await client.disconnect()

    # Build final output with channel metadata at root + messages list
    channel_link = None
    if getattr(entity, "username", None):
        channel_link = f"https://t.me/{entity.username}"

    output = messages

    try:
        # Создаем папку data если её нет
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Формируем путь для сохранения в папке data
        data_out = os.path.join(data_dir, os.path.basename(args.out))
        
        # Сохраняем только в папку data
        with open(data_out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(messages)} messages to {data_out}")

    except Exception as e:
        raise SystemExit(f"Failed to write output file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
