#!/usr/bin/env bash
# update_all.sh
# Простой набор вызовов news_parser.py для перечисленных каналов.
# Замените YOUR_API_ID и YOUR_API_HASH на реальные значения или вставьте их вручную в команду.

PY="python news_parser_advance.py"
API_ID="API_ID"
API_HASH="API_HASH"
COMMON_ARGS="--limit 200 --merge-replies --fetch-referenced --include-referenced --drop-media-only --dedupe"

# rian_ru
$PY --api-id $API_ID --api-hash $API_HASH --channel @rian_ru --out rian_ru_messages.json $COMMON_ARGS

# minzdravru
$PY --api-id $API_ID --api-hash $API_HASH --channel @minzdravru --out minzdravru_messages.json $COMMON_ARGS

# tass_agency
$PY --api-id $API_ID --api-hash $API_HASH --channel @tass_agency --out tass_agency_messages.json $COMMON_ARGS

# vestiru
$PY --api-id $API_ID --api-hash $API_HASH --channel @vestiru --out vestiru_messages.json $COMMON_ARGS

# minsportrf
$PY --api-id $API_ID --api-hash $API_HASH --channel @minsportrf --out minsportrf_messages.json $COMMON_ARGS

# mincifry
$PY --api-id $API_ID --api-hash $API_HASH --channel @mincifry --out mincifry_messages.json $COMMON_ARGS

# government_rus
$PY --api-id $API_ID --api-hash $API_HASH --channel @government_rus --out government_rus_messages.json $COMMON_ARGS

# dumainfo
$PY --api-id $API_ID --api-hash $API_HASH --channel @dumainfo --out dumainfo_messages.json $COMMON_ARGS
