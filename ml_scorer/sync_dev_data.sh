#!/usr/bin/env bash
# Marker-gated DEV data sync for the standalone V3 scorer.
#
# The TradeWave app box proves that its US/ETF EOD refresh is complete before
# this script incrementally pulls the four price-data directories used by V3.
# The scorer is restarted only after the source marker validates, rsync succeeds,
# all 26 shared context series have the completed US session, and the source
# marker is unchanged. No source-side or scorer-side files are deleted.
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ABORT: sync_dev_data.sh must run as root" >&2
  exit 1
fi

source_host=${ML_SCORER_DEV_DATA_SOURCE:-}
case "$source_host" in
  root@192.168.1.176) ;;
  *)
    echo "ABORT: ML_SCORER_DEV_DATA_SOURCE must be root@192.168.1.176" >&2
    exit 1
    ;;
esac

scorer_root=/home/flask/ml_scorer
data_root=${ML_SCORER_DATA_DIR:-/home/flask/data}
python_bin=/home/flask/venv/bin/python
source_release=/home/flask/.tw2-app-current
source_data=/home/flask/data
state_dir=/var/lib/ml_scorer
state_file=$state_dir/dev-data-generation.json

[[ -x "$python_bin" ]] || { echo "ABORT: scorer Python is unavailable" >&2; exit 1; }
[[ -d "$scorer_root" ]] || { echo "ABORT: scorer release is unavailable" >&2; exit 1; }
[[ -d "$data_root/csv" ]] || { echo "ABORT: scorer CSV root is unavailable" >&2; exit 1; }

exec 9>/run/lock/ml-scorer-dev-data-sync.lock
if ! flock -n 9; then
  echo "DEV scorer data sync already running; leaving it in control."
  exit 0
fi

sync_tmp=$(mktemp -d /var/tmp/ml-scorer-dev-sync.XXXXXX)
cleanup() {
  find "$sync_tmp" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
  rmdir "$sync_tmp" 2>/dev/null || true
}
trap cleanup EXIT

ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
marker_before=$sync_tmp/update_status.before.json
marker_after=$sync_tmp/update_status.after.json
readiness_module=$sync_tmp/eod_readiness.py

if ! ssh "${ssh_options[@]}" "$source_host" \
  'test -r /var/lib/tradewave/eod/update_status.json && cat /var/lib/tradewave/eod/update_status.json' \
  >"$marker_before"; then
  echo "DEV scorer data sync deferred: authoritative TradeWave marker is absent."
  exit 0
fi
if ! scp -q "${ssh_options[@]}" \
  "$source_host:$source_release/data_updater/eod_readiness.py" \
  "$readiness_module"; then
  echo "DEV scorer data sync deferred: readiness validator is unavailable." >&2
  exit 0
fi

marker_identity=$(
  "$python_bin" - "$marker_before" "$readiness_module" <<'PY'
import datetime as dt
import importlib.util
import json
import sys

marker_path, module_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("tw2_eod_readiness", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(marker_path, encoding="utf-8") as handle:
    marker = json.load(handle)
now = dt.datetime.now(dt.timezone.utc)
completed = module.latest_completed_us_equity_session(now).isoformat()
target = module.target_table_date(now).isoformat()
if not module.validate_success_marker(
    marker,
    expected_target_table_date=target,
    expected_completed_session=completed,
):
    raise SystemExit("authoritative marker does not validate for the current session")
print(
    marker["generation_fingerprint"],
    marker["readiness_fingerprint"],
    completed,
    target,
    sep="\t",
)
PY
) || {
  echo "DEV scorer data sync deferred: authoritative marker is not current." >&2
  exit 0
}
IFS=$'\t' read -r generation_fingerprint readiness_fingerprint completed_session target_date \
  <<<"$marker_identity"

if [[ -f "$state_file" ]] && STATE_FILE="$state_file" \
  EXPECTED_GENERATION="$generation_fingerprint" \
  "$python_bin" -c \
  'import json, os; value=json.load(open(os.environ["STATE_FILE"], encoding="utf-8")); raise SystemExit(0 if value.get("generation_fingerprint") == os.environ["EXPECTED_GENERATION"] else 1)'; then
  echo "DEV scorer data already synchronized for generation ${generation_fingerprint:0:12}."
  exit 0
fi

for market in US ETF INDX COMM; do
  install -d -o flask -g flask -m 0755 "$data_root/csv/$market"
  rsync -a --delay-updates \
    "$source_host:$source_data/csv/$market/" \
    "$data_root/csv/$market/"
done

ssh "${ssh_options[@]}" "$source_host" \
  'test -r /var/lib/tradewave/eod/update_status.json && cat /var/lib/tradewave/eod/update_status.json' \
  >"$marker_after"
"$python_bin" - "$marker_before" "$marker_after" "$readiness_module" \
  "$completed_session" "$target_date" <<'PY'
import importlib.util
import json
import sys

before_path, after_path, module_path, completed, target = sys.argv[1:]
spec = importlib.util.spec_from_file_location("tw2_eod_readiness", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(before_path, encoding="utf-8") as handle:
    before = json.load(handle)
with open(after_path, encoding="utf-8") as handle:
    after = json.load(handle)
if not module.validate_success_marker(
    after,
    expected_target_table_date=target,
    expected_completed_session=completed,
):
    raise SystemExit("authoritative marker changed to an invalid generation")
if before.get("readiness_fingerprint") != after.get("readiness_fingerprint"):
    raise SystemExit("authoritative marker changed during scorer data sync")
PY

SCORER_ROOT="$scorer_root" EXPECTED_SESSION="$completed_session" \
  sudo -u flask env \
    ML_SCORER_SKIP_INIT=1 \
    ML_SCORER_DATA_DIR="$data_root" \
    SCORER_ROOT="$scorer_root" \
    EXPECTED_SESSION="$completed_session" \
    "$python_bin" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["SCORER_ROOT"])
import metadata

expected = os.environ["EXPECTED_SESSION"]
wrong = [
    (name, metadata._tail_date(path))
    for name, path in metadata._required_context_sources()
    if metadata._tail_date(path) != expected
]
if wrong:
    raise SystemExit(f"shared scorer sources are not aligned to {expected}: {wrong}")
PY

systemctl restart ml_scorer
health_file=$sync_tmp/health.json
for _ in {1..45}; do
  if curl -fsS http://127.0.0.1:7675/health >"$health_file"; then
    break
  fi
  sleep 1
done
HEALTH_FILE="$health_file" EXPECTED_SESSION="$completed_session" \
  "$python_bin" -c \
  'import json, os; health=json.load(open(os.environ["HEALTH_FILE"], encoding="utf-8")); assert health["status"] == "ok"; assert health["feature_count"] == 62; assert health["feature_schema_version"] == "v3-62"; assert health["context_data_complete"] is True; assert health["data_as_of"] == os.environ["EXPECTED_SESSION"]'

install -d -o root -g root -m 0755 "$state_dir"
STATE_OUTPUT=$sync_tmp/dev-data-generation.json \
  GENERATION="$generation_fingerprint" \
  READINESS="$readiness_fingerprint" \
  SESSION="$completed_session" \
  TARGET_DATE="$target_date" \
  SOURCE_HOST="$source_host" \
  "$python_bin" - <<'PY'
import datetime as dt
import json
import os

payload = {
    "generation_fingerprint": os.environ["GENERATION"],
    "readiness_fingerprint": os.environ["READINESS"],
    "completed_session": os.environ["SESSION"],
    "target_table_date": os.environ["TARGET_DATE"],
    "source_host": os.environ["SOURCE_HOST"],
    "synchronized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
}
with open(os.environ["STATE_OUTPUT"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
install -o root -g root -m 0644 "$sync_tmp/dev-data-generation.json" "$state_file"

echo "DEV scorer data synchronized: session=$completed_session target=$target_date generation=${generation_fingerprint:0:12}"
