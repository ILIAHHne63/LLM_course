#!/usr/bin/env python3
"""
tg_export.py
Export messages from a Telegram channel into a JSON file.

Usage examples:
  python tg_export.py --api-id 12345 --api-hash abcd... --channel @example_channel --limit 200 --out messages.json
  python tg_export.py --session mysession --channel https://t.me/example_channel --limit 50 --download-media --media-dir media
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

def message_to_dict(msg):
    # Basic safe extraction of message fields
    data = {
        "id": msg.id,
        "date": msg.date.isoformat() if msg.date else None,
        "text": msg.message or "",
        "raw_text_len": len(msg.message or ""),
        "views": getattr(msg, "views", None),
        "forwards": getattr(msg, "forwards", None),
        "reply_to_msg_id": getattr(msg.reply_to_msg_id, "reply_to_msg_id", msg.reply_to_msg_id) if getattr(msg, "reply_to_msg_id", None) else None,
        "is_reply": bool(getattr(msg, "reply_to_msg_id", None)),
        "sender_id": getattr(msg, "sender_id", None),
    }
    # Media presence
    data["has_media"] = bool(getattr(msg, "media", None))
    return data

async def fetch_messages(client, channel, limit, download_media, media_dir):
    entity = await client.get_entity(channel)
    out = []
    # If limit == 0, iter_messages yields all messages (be careful)
    fetch_limit = None if (limit == 0) else limit

    # Ensure media dir exists
    if download_media:
        os.makedirs(media_dir, exist_ok=True)

    async for msg in client.iter_messages(entity, limit=fetch_limit):
        try:
            item = message_to_dict(msg)
            # If there is media, optionally download
            if item["has_media"]:
                media_info = {"downloaded": None, "media_type": None}
                # try to infer type
                try:
                    kind = msg.media.__class__.__name__
                    media_info["media_type"] = kind
                except Exception:
                    media_info["media_type"] = "unknown"

                if download_media:
                    # file name pattern: <channelid>_<msgid>_<origfilename or extension>
                    base = f"{entity.id}_{msg.id}"
                    try:
                        saved = await client.download_media(msg, file=os.path.join(media_dir, base))
                        # download_media can return path or bytes
                        if isinstance(saved, str):
                            media_info["downloaded"] = saved
                        else:
                            # if bytes returned, write to file
                            fname = os.path.join(media_dir, base + ".bin")
                            with open(fname, "wb") as f:
                                f.write(saved)
                            media_info["downloaded"] = fname
                    except Exception as e:
                        media_info["download_error"] = str(e)
                item["media"] = media_info
            out.append(item)
        except Exception as e:
            # don't stop on single message error
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
        messages = await fetch_messages(client, args.channel, args.limit, args.download_media, args.media_dir)
    except RPCError as e:
        raise SystemExit(f"Telegram RPC error: {e}")
    except Exception as e:
        raise SystemExit(f"Error fetching messages: {e}")
    finally:
        await client.disconnect()

    # Write output JSON
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(messages)} messages to {args.out}")
    except Exception as e:
        raise SystemExit(f"Failed to write output file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
