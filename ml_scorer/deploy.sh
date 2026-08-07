#!/bin/bash
# Build and activate a fresh commit-named ml_scorer release on the dev server.
#
#   ML_SCORER_ARTIFACT_SOURCE=/path/to/verified/release ./deploy.sh deploy
#   ./deploy.sh check
#   ./deploy.sh rollback /home/flask/.ml-scorer-releases/<release>
#
# The active path is a symlink. Never copy files through it: doing so mutates
# the prior release and destroys rollback integrity.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
RELEASE_ROOT=${ML_SCORER_RELEASE_ROOT:-/home/flask/.ml-scorer-releases}
ACTIVE_LINK=${ML_SCORER_ACTIVE_LINK:-/home/flask/ml_scorer}
PREVIOUS_LINK=${ML_SCORER_PREVIOUS_LINK:-/home/flask/ml_scorer.previous}
DATA_DIR=${ML_SCORER_DATA_DIR:-/home/flask/data}
PYTHON=${ML_SCORER_PYTHON:-/home/flask/venv/bin/python}
SERVICE=${ML_SCORER_SERVICE:-ml_scorer.service}
EXPECT_RUN=${EXPECT_RUN:-20260802,20260803}
PREFLIGHT_N=${ML_SCORER_PREFLIGHT_N:-80}
MODE=${1:-check}

STAGE_ROOT=
NEXT_LINK=

cleanup() {
  if [ -n "$NEXT_LINK" ] && [ -L "$NEXT_LINK" ]; then
    rm -f -- "$NEXT_LINK"
  fi
  if [ -n "$STAGE_ROOT" ] && [ -d "$STAGE_ROOT" ]; then
    rm -rf -- "$STAGE_ROOT"
  fi
}
trap cleanup EXIT

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_root() {
  [ "$(id -u)" -eq 0 ] || fail 'deploy and rollback must run as root'
}

resolve_release() {
  local path=$1
  readlink -f -- "$path" 2>/dev/null || true
}

validate_release_path() {
  local release=$1
  [ -n "$release" ] || fail 'release path is empty'
  case "$release" in
    "$RELEASE_ROOT"/*) ;;
    *) fail "release must be under $RELEASE_ROOT: $release" ;;
  esac
  [ -d "$release" ] || fail "release directory does not exist: $release"
  [ -f "$release/app.py" ] || fail "release has no app.py: $release"
}

activate_link() {
  local release=$1
  validate_release_path "$release"
  NEXT_LINK="${ACTIVE_LINK}.next.$$"
  if [ -e "$NEXT_LINK" ] || [ -L "$NEXT_LINK" ]; then
    fail "temporary link exists: $NEXT_LINK"
  fi
  ln -s "$release" "$NEXT_LINK"
  mv -Tf -- "$NEXT_LINK" "$ACTIVE_LINK"
  NEXT_LINK=
}

wait_for_health() {
  local socket="$ACTIVE_LINK/ml_scorer.sock"
  local _attempt
  for _attempt in $(seq 1 60); do
    if curl -fsS --unix-socket "$socket" http://localhost/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

validate_route_map() {
  local release=$1
  (
    cd "$release"
    sudo -u flask env \
      PYTHONDONTWRITEBYTECODE=1 \
      ML_SCORER_DATA_DIR="$DATA_DIR" \
      ML_SCORER_SKIP_INIT=1 \
      "$PYTHON" - <<'PY'
from app import app

required = {'/health', '/metadata', '/tiers', '/score', '/score/context', '/select'}
actual = {rule.rule for rule in app.url_map.iter_rules()}
missing = sorted(required - actual)
if missing:
    raise SystemExit(f'missing routes: {missing}')
print('route map: ok')
PY
  )
}

validate_live_contract() {
  local socket="$ACTIVE_LINK/ml_scorer.sock"

  curl -fsS --unix-socket "$socket" http://localhost/health | "$PYTHON" -c '
import json, sys
p = json.load(sys.stdin)
assert p["status"] == "ok", p
assert p["feature_count"] == 62, p
assert p["feature_schema_version"] == "v3-62", p
assert p["context_schema_version"] == "duration-comparison-context-v5", p
print("health contract: ok")
'

  curl -fsS --unix-socket "$socket" http://localhost/metadata | "$PYTHON" -c '
import json, sys
p = json.load(sys.stdin)
m = p["metadata"]
assert m["feature_schema_version"] == "v3-62", p
assert m["context_schema_version"] == "duration-comparison-context-v5", p
assert p["model_manifest"], p
print("metadata contract: ok")
'

  curl -fsS --max-time 300 --unix-socket "$socket" \
    -H 'Content-Type: application/json' \
    -d '{"symbol":"AAPL","date":"2026-08-05","daysOut":29,"direction":"l"}' \
    http://localhost/score | "$PYTHON" -c '
import json, math, sys
p = json.load(sys.stdin)
r = p["results"][0]
assert r["tier"] == "10_30", p
assert math.isfinite(float(r["ml_score"])), p
print("legacy score contract: ok")
'

  curl -fsS --max-time 300 --unix-socket "$socket" \
    -H 'Content-Type: application/json' \
    -d '{"resource_id":"2","symbol":"AAPL","date":"2026-08-05","calendar_days":30,"direction":"l","years":"20","partial":null}' \
    http://localhost/score/context | "$PYTHON" -c '
import json, math, sys
p = json.load(sys.stdin)
r = p["results"][0]
assert r["status"] == "ok", p
assert r["calendar_days"] == 30 and r["daysOut"] == 29, p
assert r["tier"] == "10_30" and r["pattern_recalculated"] is True, p
for name in ("ml_score", "win_prob", "pred_return", "pred_mfe"):
    assert math.isfinite(float(r[name])), p
print("context score contract: ok")
'
}

run_source_tests() {
  (
    cd "$REPO_ROOT"
    env \
      PYTHONDONTWRITEBYTECODE=1 \
      ML_SCORER_DATA_DIR="$DATA_DIR" \
      ML_SCORER_SKIP_INIT=1 \
      "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
  )
}

run_candidate_preflight() {
  local release=$1
  (
    cd "$release"
    sudo -u flask env \
      PYTHONDONTWRITEBYTECODE=1 \
      ML_SCORER_DATA_DIR="$DATA_DIR" \
      "$PYTHON" preflight.py --expect-run "$EXPECT_RUN" --skip-live
    sudo -u flask env \
      PYTHONDONTWRITEBYTECODE=1 \
      ML_SCORER_DATA_DIR="$DATA_DIR" \
      "$PYTHON" preflight.py --expect-run "$EXPECT_RUN" --n "$PREFLIGHT_N"
  )
}

case "$MODE" in
  check)
    active=$(resolve_release "$ACTIVE_LINK")
    validate_release_path "$active"
    printf 'active release: %s\n' "$active"
    if [ -f "$active/DEPLOY.txt" ]; then
      sed -n '1,20p' "$active/DEPLOY.txt"
    fi
    validate_route_map "$active"
    ;;

  rollback)
    require_root
    target=${2:-$(resolve_release "$PREVIOUS_LINK")}
    validate_release_path "$target"
    current=$(resolve_release "$ACTIVE_LINK")
    activate_link "$target"
    systemctl restart "$SERVICE"
    if ! wait_for_health || ! validate_live_contract; then
      if [ -n "$current" ] && [ -d "$current" ]; then
        activate_link "$current"
        systemctl restart "$SERVICE"
      fi
      fail "rollback target failed validation: $target"
    fi
    printf 'rolled back to: %s\n' "$target"
    ;;

  deploy)
    require_root
    [ -x "$PYTHON" ] || fail "Python interpreter not executable: $PYTHON"
    mkdir -p "$RELEASE_ROOT"

    status=$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)
    [ -z "$status" ] || fail 'git worktree must be clean before deployment'
    sha=$(git -C "$REPO_ROOT" rev-parse HEAD)
    release="$RELEASE_ROOT/$sha"
    [ ! -e "$release" ] || fail "release already exists and will not be overwritten: $release"

    for path in \
      ml_scorer/app.py \
      ml_scorer/config.py \
      ml_scorer/context_contract.py \
      ml_scorer/feature_engine.py \
      ml_scorer/metadata.py \
      ml_scorer/preflight.py \
      ml_scorer/training_bounds.json \
      ml_scorer/sync_dev_data.sh \
      tests/test_context_scoring.py; do
      git -C "$REPO_ROOT" cat-file -e "$sha:$path" || fail "commit omits required file: $path"
    done

    artifact_source=${ML_SCORER_ARTIFACT_SOURCE:-$(resolve_release "$ACTIVE_LINK")}
    validate_release_path "$artifact_source"
    [ -d "$artifact_source/models" ] || fail "artifact source has no models: $artifact_source"
    [ -d "$artifact_source/calibration" ] || fail "artifact source has no calibration: $artifact_source"

    rollback_target=${ML_SCORER_ROLLBACK_TARGET:-$(resolve_release "$ACTIVE_LINK")}
    validate_release_path "$rollback_target"

    run_source_tests

    STAGE_ROOT=$(mktemp -d "$RELEASE_ROOT/.stage-${sha}.XXXXXX")
    chmod 0755 "$STAGE_ROOT"
    git -C "$REPO_ROOT" archive "$sha" ml_scorer | tar -x -C "$STAGE_ROOT"
    candidate="$STAGE_ROOT/ml_scorer"
    cp -a "$artifact_source/models" "$candidate/models"
    rm -rf -- "$candidate/calibration"
    cp -a "$artifact_source/calibration" "$candidate/calibration"
    sed -i "s|python3.12|$PYTHON|g" "$candidate/nightly.sh"

    {
      printf 'commit=%s\n' "$sha"
      printf 'branch=%s\n' "$(git -C "$REPO_ROOT" branch --show-current)"
      printf 'built_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'artifact_source=%s\n' "$artifact_source"
      printf 'rollback_target=%s\n' "$rollback_target"
    } > "$candidate/DEPLOY.txt"

    chown -R root:www-data "$candidate"
    find "$candidate" -type d -exec chmod 0555 {} +
    find "$candidate" -type f -exec chmod 0444 {} +
    chmod 0555 \
      "$candidate/deploy.sh" \
      "$candidate/nightly.sh" \
      "$candidate/opp_to_parquet.py" \
      "$candidate/preflight.py" \
      "$candidate/sync_dev_data.sh" \
      "$candidate/warmup_cache.py"
    chmod 3775 "$candidate"

    "$PYTHON" -m compileall -q "$candidate"
    validate_route_map "$candidate"
    cmp \
      <(cd "$artifact_source" && sha256sum models/* calibration/*.json) \
      <(cd "$candidate" && sha256sum models/* calibration/*.json)
    run_candidate_preflight "$candidate"

    mv -- "$candidate" "$release"
    rmdir -- "$STAGE_ROOT"
    STAGE_ROOT=

    old_release=$(resolve_release "$ACTIVE_LINK")
    activate_link "$release"
    systemctl restart "$SERVICE"

    if ! wait_for_health || ! validate_live_contract; then
      printf 'Activation validation failed. Restoring %s\n' "$rollback_target" >&2
      activate_link "$rollback_target"
      systemctl restart "$SERVICE"
      wait_for_health || true
      fail "release failed after activation: $release"
    fi

    previous_next="${PREVIOUS_LINK}.next.$$"
    ln -s "$rollback_target" "$previous_next"
    mv -Tf -- "$previous_next" "$PREVIOUS_LINK"

    printf 'previous active release: %s\n' "$old_release"
    printf 'active release: %s\n' "$release"
    ;;

  *)
    fail 'usage: deploy.sh {check|deploy|rollback [release_path]}'
    ;;
esac
