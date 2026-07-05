# Scoreboard Mini App — deploy

A Telegram Mini App that shows whether the drafter is getting better at
mimicking the owner's voice over time. Three layers:

1. **Metrics API** — `GET /scoreboard` on the Mirror approve-service
   (`agent/approve_service.py` → `agent/scoreboard.py`). Computes, from the
   `feedback` table: approve-without-edit rate, edit-magnitude trend, and
   style-similarity trend (draft style embedding vs the owner's style centroid),
   plus totals. Owner-gated (see Auth).
2. **Mini App** — `webapp/scoreboard.html`, one self-contained file (inline
   CSS/JS, hand-rolled SVG charts, only `telegram-web-app.js` external). Served
   by the same service at `/` and `/scoreboard.html`.
3. **HTTPS + bot button** — a Cloudflare tunnel exposes the scoreboard routes;
   the bot's chat menu button opens the Mini App.

## Auth (owner-only)
`GET /scoreboard` accepts EITHER:
- `X-Mirror-Secret: <MIRROR_APPROVE_SECRET>` (local testing), OR
- `X-Telegram-Init-Data: <initData>` — validated by HMAC against
  `MIRROR_WEBAPP_BOT_TOKEN` (the `WebAppData` scheme) AND required to carry
  `user.id == MIRROR_OWNER_USER_ID`, with a fresh `auth_date`.

Set in the service `.env` (never commit real values):
```
MIRROR_WEBAPP_BOT_TOKEN=<token of the bot hosting the menu button>
MIRROR_OWNER_USER_ID=<owner numeric telegram id>
MIRROR_APPROVE_SECRET=<already set for /draft,/decide>
```

## HTTPS (Cloudflare tunnel)
The service binds loopback `127.0.0.1:8791`. Expose ONLY the scoreboard/html/
health paths (the send-as-owner `/draft`,`/decide` stay off the public router
and are secret-gated regardless). Reuse an existing named tunnel:

```bash
# 1. DNS route (uses the tunnel cert — no separate API token):
cloudflared tunnel route dns <TUNNEL_NAME> scoreboard.<YOUR_DOMAIN>

# 2. Add to ~/.cloudflared/config.yml ingress (BEFORE the catch-all):
#   - hostname: scoreboard.<YOUR_DOMAIN>
#     path: ^/($|scoreboard|health)
#     service: http://127.0.0.1:8791
#   - hostname: scoreboard.<YOUR_DOMAIN>
#     service: http_status:404

# 3. Restart the tunnel:
sudo systemctl restart <TUNNEL_SERVICE>
```

## Bot menu button
Register the Mini App as the owner's chat menu button on the hosting bot
(no bot restart needed):

```bash
curl -s -X POST "https://api.telegram.org/bot<BOT_TOKEN>/setChatMenuButton" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": <OWNER_ID>,
       "menu_button": {"type":"web_app","text":"📊 Scoreboard",
         "web_app":{"url":"https://scoreboard.<YOUR_DOMAIN>/"}}}'
```

The owner then opens it from the menu button (bottom-left) in the bot chat.
Because the button is scoped to the owner's chat_id, no one else sees it, and
the API enforces owner-only regardless.

## Verify
```bash
curl -s -H "X-Mirror-Secret: $SECRET" http://127.0.0.1:8791/scoreboard | jq .
curl -s -o /dev/null -w '%{http_code}\n' https://scoreboard.<YOUR_DOMAIN>/   # 200
curl -s -o /dev/null -w '%{http_code}\n' https://scoreboard.<YOUR_DOMAIN>/scoreboard  # 401 (no auth)
```
