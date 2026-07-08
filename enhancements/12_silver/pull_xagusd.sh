#!/bin/zsh
# Task 27 Track B — XAGUSD M1 acquisition v2 (post rate-limit lesson).
# SERIAL per-year chunks with the CLI's own throttling (-bs 3 -bp 2000 -r 5
# -rp 3000), row-validated after each chunk, 20s between chunks, and an
# initial probe loop that waits out any active throttle (tiny 2-day pull
# every 10 min, up to 12 attempts). Resumable: non-empty chunks are skipped.
# Usage: zsh enhancements/12_silver/pull_xagusd.sh
set -u
cd "$(dirname "$0")/../.."
DIR=data/dukascopy_raw
mkdir -p "$DIR"
THROTTLE_FLAGS=(-bs 3 -bp 2000 -r 5 -rp 3000)

probe() {
  rm -rf /tmp/xag_probe
  npx --yes dukascopy-node -i xagusd -from 2024-06-03 -to 2024-06-05 -t m1 -p bid \
      -f csv "${THROTTLE_FLAGS[@]}" -dir /tmp/xag_probe >/dev/null 2>&1
  f=(/tmp/xag_probe/xagusd-m1-bid-*.csv(N))
  [ ${#f} -ge 1 ] && [ "$(wc -l < "${f[1]}" | tr -d ' ')" -gt 100 ]
}

echo "=== throttle probe ==="
ok=0
for attempt in $(seq 1 12); do
  if probe; then echo "[probe] OK on attempt $attempt"; ok=1; break; fi
  echo "[probe] throttled (attempt $attempt/12) — waiting 600s"
  sleep 600
done
[ "$ok" -eq 1 ] || { echo "PROBE NEVER SUCCEEDED — aborting"; exit 1; }

pull_one() {
  price=$1; from=$2; to=$3
  # any existing non-empty chunk covering this start counts as done (the CLI
  # renames when data starts later than requested, e.g. 2003)
  existing=($DIR/xagusd-m1-$price-${from%%-01-01}*.csv(N))
  for e in $existing; do
    [ "$(wc -l < "$e" | tr -d ' ')" -gt 100 ] && { echo "[skip] $e"; return 0; }
  done
  for try in 1 2 3; do
    npx --yes dukascopy-node -i xagusd -from "$from" -to "$to" -t m1 -p "$price" \
        -f csv -v true "${THROTTLE_FLAGS[@]}" -dir "$DIR" >/dev/null 2>&1
    # validate: a matching non-trivial file for this chunk's start year must exist
    got=""
    for e in $DIR/xagusd-m1-$price-*.csv(N); do
      base=${e:t}
      case "$base" in xagusd-m1-$price-${from:0:4}*) got="$e";; esac
    done
    if [ -n "$got" ] && [ "$(wc -l < "$got" | tr -d ' ')" -gt 100 ]; then
      echo "[done] ${got:t} ($(wc -l < "$got" | tr -d ' ') rows, try $try)"
      return 0
    fi
    rm -f $DIR/xagusd-m1-$price-${from:0:4}*.csv(N) 2>/dev/null
    echo "[retry] $price $from try $try failed — backoff $((try*120))s"
    sleep $((try*120))
  done
  echo "[FAIL] $price $from after 3 tries"
  return 1
}

fails=0
for y in $(seq 2003 2025); do
  pull_one bid "$y-01-01" "$((y+1))-01-01" || fails=$((fails+1))
  sleep 20
done
pull_one bid 2026-01-01 2026-07-08 || fails=$((fails+1)); sleep 20
for y in 2023 2024 2025; do
  pull_one ask "$y-01-01" "$((y+1))-01-01" || fails=$((fails+1))
  sleep 20
done
pull_one ask 2026-01-01 2026-07-08 || fails=$((fails+1))

n_bid=$(ls $DIR/xagusd-m1-bid-*.csv 2>/dev/null | wc -l | tr -d ' ')
n_ask=$(ls $DIR/xagusd-m1-ask-*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "PULL COMPLETE: bid $n_bid/24, ask $n_ask/4, fails $fails"
[ "$fails" -eq 0 ] && [ "$n_bid" -eq 24 ] && [ "$n_ask" -eq 4 ] && echo "ALL CHUNKS OK" || echo "CHUNKS MISSING — rerun"
