#!/usr/bin/env bash
# Telegram notifier for the RL run. Runs on n82.
#
#   per step   -> message with step time, reward, running mean, and health
#   frozen run -> repeated CALL via CallMeBot (Telegram cannot place calls
#                 itself; CallMeBot rings your Telegram account)
#
# Credentials come from the environment so nothing secret lands in git:
#   TG_BOT_TOKEN   from @BotFather
#   TG_CHAT_ID     from @userinfobot ("Id")
#   CALLMEBOT_KEY  optional; from @CallMeBot_txtbot, enables the phone call
#
# Freeze definition: the TRAINER step counter has not advanced and the
# ORCHESTRATOR batch counter has not advanced for FREEZE_MIN minutes. Both must
# be stuck -- a slow batch alone is normal and must not page anyone.
set -uo pipefail

RUN=${RUN:-x22}
RUN_NAME=${RUN_NAME:-run_$RUN}
INTERVAL=${INTERVAL:-60}
FREEZE_MIN=${FREEZE_MIN:-15}
TG_BOT_TOKEN=${TG_BOT_TOKEN:-}
TG_CHAT_ID=${TG_CHAT_ID:-}
CALLMEBOT_KEY=${CALLMEBOT_KEY:-}
STATE=/data/glm52-rl/notify.state

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

send() { # text
  [ -z "$TG_BOT_TOKEN" ] && { log "TG_BOT_TOKEN unset; would send: $1"; return; }
  curl -s -m 20 -o /dev/null -X POST \
    "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
    --data-urlencode "chat_id=${TG_CHAT_ID}" \
    --data-urlencode "text=$1" >/dev/null 2>&1
}

call() { # text -- rings the phone, for freezes only
  [ -z "$CALLMEBOT_KEY" ] && { log "CALLMEBOT_KEY unset; no call placed"; return; }
  curl -s -m 25 -o /dev/null -G "https://api.callmebot.com/start.php" \
    --data-urlencode "user=@${TG_CHAT_ID}" \
    --data-urlencode "text=$1" \
    --data-urlencode "lang=en-US-Standard-B" \
    --data-urlencode "rpt=2" >/dev/null 2>&1
}

tsteps() { sudo docker logs glm52-decensor-trainer 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -ac "SUCCESS Step"; }
tlast()  { sudo docker logs glm52-decensor-trainer 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -a "SUCCESS Step" | tail -1; }
rewards(){ sudo docker logs glm52-decensor-orchestrator 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -a "SUCCESS Step" | grep -oE "Reward [0-9.]+" | awk '{print $2}'; }
batchn() { sudo docker logs --tail 5 glm52-decensor-orchestrator 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "batch [0-9]+/" | tail -1 | grep -oE "[0-9]+"; }

last_step=$(cat "$STATE" 2>/dev/null || echo 0)
last_batch=-1
stuck=0
frozen_notified=0
log "notifier started (run=$RUN interval=${INTERVAL}s freeze=${FREEZE_MIN}m)"

while true; do
  n=$(tsteps 2>/dev/null || echo 0)
  b=$(batchn 2>/dev/null || echo -1)

  if [ "${n:-0}" -gt "${last_step:-0}" ] 2>/dev/null; then
    line=$(tlast | grep -oE "Step [0-9]+ \| *[0-9hms ]+\| Loss [^|]*\| Entropy [^|]*\| Grad. Norm [^|]*")
    rs=$(rewards)
    mean=$(echo "$rs" | awk 'NF{s+=$1;c++} END{if(c)printf "%.4f (n=%d)",s/c,c; else print "n/a"}')
    recent=$(echo "$rs" | tail -5 | tr '\n' ' ')
    send "GLM-5.2 ${RUN} — trainer step ${n}/200
${line}
mean reward: ${mean}
last 5: ${recent}"
    log "notified step $n"
    last_step=$n; echo "$n" > "$STATE"
    stuck=0; frozen_notified=0
  fi

  # Frozen only if BOTH the step counter and the batch counter are static.
  if [ "$b" = "$last_batch" ] && [ "${n:-0}" -eq "${last_step:-0}" ]; then
    stuck=$((stuck + INTERVAL))
  else
    stuck=0; frozen_notified=0
  fi
  last_batch=$b

  if [ "$stuck" -ge $((FREEZE_MIN * 60)) ] && [ "$frozen_notified" -eq 0 ]; then
    msg="GLM-5.2 ${RUN} FROZEN: no trainer step and no batch progress for ${FREEZE_MIN} min (step ${n}, batch ${b})"
    log "FREEZE detected"
    send "$msg"
    call "GLM training run frozen. No progress for ${FREEZE_MIN} minutes."
    frozen_notified=1
    stuck=0
  fi

  sleep "$INTERVAL"
done
