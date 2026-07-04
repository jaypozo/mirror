#!/usr/bin/env bash
#
# Mirror LIVE approve-UI service runner.
#
# The Telethon USER session (.telethon/mirror) is single-writer SQLite: only ONE
# process may hold it. The standalone history backfill drip
# (ingest/backfill_drip.sh) holds it during corpus building. So `start` here
# STOPS the standalone drip FIRST, then launches the live service as the single
# owner of the user session. If you want history backfill to continue, run the
# live service with MIRROR_RESUME_BACKFILL=1 — it resumes the drip as a
# background task INSIDE the same client (no second concurrent session).
#
# Usage:
#   agent/service_run.sh start    # stop standalone drip, then start the service
#   agent/service_run.sh stop     # stop the service (SIGTERM; clean shutdown)
#   agent/service_run.sh status   # pid + tail of the log
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PY="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/service.log"
PID_FILE="$LOG_DIR/service.pid"
DRIP="$PROJECT_DIR/ingest/backfill_drip.sh"

export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Any python puller (pull_recent/pull/backfill) also holds the single session.
session_busy_pids() {
  pgrep -f "python.* -m ingest\.(pull_recent|pull|backfill)" || true
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

    # 2) Belt-and-braces: kill any lingering puller still holding the session.
    lingering="$(session_busy_pids)"
    if [[ -n "$lingering" ]]; then
      echo "waiting for lingering pullers to release the session: $lingering"
      # shellcheck disable=SC2086
      kill $lingering 2>/dev/null || true
      for _ in $(seq 1 20); do
        [[ -z "$(session_busy_pids)" ]] && break
        sleep 0.5
      done
    fi
    if [[ -n "$(session_busy_pids)" ]]; then
      echo "refusing to start: a puller still holds the Telethon session:"
      session_busy_pids
      exit 1
    fi

    # 3) Start the live service, detached + low priority + logged.
    setsid nice -n 5 "$PY" -m agent.service >> "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "started Mirror service pid $(cat "$PID_FILE")"
    echo "log:  $LOG_FILE"
    echo "stop: $0 stop"
    if [[ -z "${MIRROR_BOT_TOKEN:-}" ]] && ! grep -q '^MIRROR_BOT_TOKEN=..*' .env 2>/dev/null; then
      echo "NOTE: MIRROR_BOT_TOKEN is empty — service runs but the approve UI is"
      echo "      OFFLINE (nothing gets sent). Add MIRROR_BOT_TOKEN to .env and"
      echo "      re-run '$0 stop && $0 start' to go live."
    fi
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
