#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"текст запроса\" [limit]" >&2
  exit 1
fi

QUERY="$1"
LIMIT="${2:-5}"

curl -s -X POST http://localhost:8000/news/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"${QUERY}\", \"limit\": ${LIMIT}}" \
  > /dev/null

python scripts/show_summary.py
