#!/usr/bin/env bash
#
# Overnight backfill drip for Mirror.
#
# Runs the downward history backfill (ingest.backfill — walks messages OLDER
# than what's already stored, per chat, until exhausted) as a low-priority
# background process with very conservative pacing so it slowly fills the corpus
# overnight without risking the owner's account.
#
# Why ingest.backfill and not ingest.pull: the recent-window pull already set
# each chat's forward cursor (sync_state.last_message_id) to the newest message,
# so ingest.pull (min_id=cursor) would fetch nothing older. ingest.backfill
# fills the older direction instead. Fully resumable: the messages table itself
# is the cursor, so kill it any time and re-run; it never re-pulls what landed.
#
# NOTE: Telegram takeout is currently on a ~24h init cooldown (prior attempts),
# so this uses the plain user client with conservative pacing + FloodWait
# handling. That is the sanctioned fallback per the plan.
#
# Usage:
#   ingest/backfill_drip.sh start   # launch nice'd background drip
#   ingest/backfill_drip.sh stop    # stop it (safe; resumable)
#   ingest/backfill_drip.sh status  # show pid + tail of log
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PY="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/backfill_drip.log"
PID_FILE="$LOG_DIR/backfill_drip.pid"

# Conservative pacing (more conservative than the pull.py defaults).
export PULL_BATCH_SIZE="${PULL_BATCH_SIZE:-100}"
export PULL_BATCH_SLEEP_SECONDS="${PULL_BATCH_SLEEP_SECONDS:-2}"
export PULL_CHAT_SLEEP_SECONDS="${PULL_CHAT_SLEEP_SECONDS:-5}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

cmd="${1:-status}"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# The Telethon .session is a single-writer SQLite file: two pullers at once
# cause "database is locked". Refuse to start if a python puller already holds
# it. Match only the actual "python -m ingest.<mod>" processes (via the venv
# python), not shells whose command line merely mentions "ingest".
session_busy() {
  pgrep -f "python.* -m ingest\.(pull_recent|pull|backfill)" >/dev/null 2>&1
}

case "$cmd" in
  start)
    if is_running; then
      echo "already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi
    if session_busy; then
      echo "refusing to start: another puller (pull_recent/pull/backfill) is"
      echo "already using the Telethon session. Stop it first to avoid a"
      echo "'database is locked' collision:"
      pgrep -af "python.* -m ingest\.(pull_recent|pull|backfill)" || true
      exit 1
    fi
    # setsid fully detaches; nice keeps it low-priority.
    setsid nice -n 15 "$PY" -m ingest.backfill >> "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "started backfill drip pid $(cat "$PID_FILE")"
    echo "log:  $LOG_FILE"
    echo "stop: $0 stop"
    ;;
  stop)
    if is_running; then
      pid="$(cat "$PID_FILE")"
      kill "$pid" 2>/dev/null || true
      echo "sent SIGTERM to pid $pid (resumable; sync_state is safe)"
      rm -f "$PID_FILE"
    else
      echo "not running"
      rm -f "$PID_FILE"
    fi
    ;;
  status)
    if is_running; then
      echo "running (pid $(cat "$PID_FILE"))"
    else
      echo "not running"
    fi
    echo "--- last 15 log lines ---"
    tail -n 15 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
    ;;
  *)
    echo "usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
