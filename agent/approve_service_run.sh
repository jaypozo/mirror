#!/usr/bin/env bash
#
# Mirror INLINE-APPROVE HTTP service runner.
#
# Companion to the fleet Telegram bots' patch F. Ttheir owns the Telethon USER
# session (.telethon/mirror, single-writer SQLite) and exposes a loopback HTTP
# API (/draft, /decide) the fleet bots POST to. It is the SOLE process that
# sends AS the owner.
#
# Because the session is single-writer, `start` STOPS the standalone history
# backfill drip FIRST (same as agent/service_run.sh), then launches this as the
# single owner. Set MIRROR_RESUME_BACKFILL=1 to resume the drip as a background
# task inside THIS client (no second concurrent session).
#
# NOTE: This does NOT run agent/service.py's Mirror-bot approve UI. The approve
# UI is the fleet bot itself (patch F). This service is headless.
#
# Usage (run from the mirror repo root, with approve_service.py at agent/):
#   agent/approve_service_run.sh start
#   agent/approve_service_run.sh stop
#   agent/approve_service_run.sh status
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PY="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/approve_service.log"
PID_FILE="$LOG_DIR/approve_service.pid"
DRIP="$PROJECT_DIR/ingest/backfill_drip.sh"

export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Anything else that holds the single Telethon session.
session_busy_pids() {
  pgrep -f "python.* -m (ingest\.(pull_recent|pull|backfill)|agent\.service)\b" || true
}

cmd="${1:-status}"

case "$cmd" in
  start)
    if is_running; then
      echo "already running (pid $(cat "$PID_FILE"))"
      exit 0
    fi

    # 1) Stop the standalone backfill drip so we own the session.
    if [[ -x "$DRIP" ]]; then
      echo "stopping standalone backfill drip (frees the user session)..."
      "$DRIP" stop || true
    fi

    # 2) Belt-and-braces: kill any lingering puller/service still holding it.
    lingering="$(session_busy_pids)"
    if [[ -n "$lingering" ]]; then
      echo "waiting for lingering session holders to release: $lingering"
      # shellcheck disable=SC2086
      kill $lingering 2>/dev/null || true
      for _ in $(seq 1 20); do
        [[ -z "$(session_busy_pids)" ]] && break
        sleep 0.5
      done
    fi
    if [[ -n "$(session_busy_pids)" ]]; then
      echo "refusing to start: something still holds the Telethon session:"
      session_busy_pids
      exit 1
    fi

    # 3) Preflight: secret must be set (the service also refuses, this is nicer).
    if [[ -z "${MIRROR_APPROVE_SECRET:-}" ]] && ! grep -q '^MIRROR_APPROVE_SECRET=..*' .env 2>/dev/null; then
      echo "refusing to start: MIRROR_APPROVE_SECRET is empty."
      echo "  Generate one:  openssl rand -hex 24"
      echo "  Put it in .env AND in the fleet bot's channel .env (same value)."
      exit 1
    fi

    # 4) Start detached + low priority + logged.
    setsid nice -n 5 "$PY" -m agent.approve_service >> "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "started Mirror approve service pid $(cat "$PID_FILE")"
    echo "log:  $LOG_FILE"
    echo "stop: $0 stop"
    ;;

  stop)
    if is_running; then
      pid="$(cat "$PID_FILE")"
      kill "$pid" 2>/dev/null || true
      echo "sent SIGTERM to pid $pid"
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
    echo "--- last 20 log lines ---"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
    ;;

  *)
    echo "usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
