#!/bin/bash
# Deploy the git workspace to the running dev ml_scorer, or check for drift.
#
#   ./deploy.sh check    compare /home/flask/ml_scorer against the git workspace
#   ./deploy.sh deploy   copy workspace -> service, restart, health-check
#
# models/ is gitignored and lives only on the box, so it is never touched.
# nightly.sh is re-patched to the venv interpreter after copying, because the
# repo version calls python3.12 (correct for prod, wrong here).
set -uo pipefail

SRC=/root/dev/ml_scorer/ml_scorer
DST=/home/flask/ml_scorer
VENV=/home/flask/venv/bin/python
FILES="app.py feature_engine.py scorer.py config.py daily_opp_selection.py
       opp_scorer.py opp_to_parquet.py warmup_cache.py
       nightly.sh requirements.txt CLAUDE.md"

mode="${1:-check}"

if [ "$mode" = "check" ]; then
  echo "workspace: $(git -C /root/dev/ml_scorer rev-parse --abbrev-ref HEAD) @ $(git -C /root/dev/ml_scorer log --oneline -1 | cut -d" " -f1)"
  drift=0
  for f in $FILES; do
    [ -f "$SRC/$f" ] || continue
    if [ "$f" = "nightly.sh" ]; then
      # compare ignoring the interpreter patch
      a=$(sed "s|$VENV|python3.12|g" "$SRC/$f" | md5sum | cut -d" " -f1)
      b=$(sed "s|$VENV|python3.12|g" "$DST/$f" 2>/dev/null | md5sum | cut -d" " -f1)
    else
      a=$(md5sum "$SRC/$f" | cut -d" " -f1)
      b=$(md5sum "$DST/$f" 2>/dev/null | cut -d" " -f1)
    fi
    [ "$a" = "$b" ] || { printf "  DRIFT  %s\n" "$f"; drift=1; }
  done
  [ $drift -eq 0 ] && echo "  no drift: deployment matches the git workspace" || echo "  ^ deployment differs from git"
  exit $drift
fi

if [ "$mode" = "deploy" ]; then
  for f in $FILES; do [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"; done
  cp "$SRC"/calibration/*.json "$DST"/calibration/ 2>/dev/null
  sed -i "s|python3.12|$VENV|g" "$DST/nightly.sh"
  chmod +x "$DST/nightly.sh" "$DST/warmup_cache.py" "$DST/opp_to_parquet.py"
  chown -R flask:flask "$DST"
  systemctl restart ml_scorer
  for i in $(seq 1 30); do
    sleep 1
    curl -sf --unix-socket "$DST/ml_scorer.sock" http://localhost/health >/dev/null 2>&1 && break
  done
  echo -n "  health: "; curl -s --unix-socket "$DST/ml_scorer.sock" http://localhost/health; echo
  echo
  echo "  --- preflight ---"
  ( cd "$DST" && ML_SCORER_DATA_DIR=/home/flask/data "$VENV" preflight.py ${EXPECT_RUN:+--expect-run "$EXPECT_RUN"} --n 80 ) || {
    echo
    echo "  *** PREFLIGHT FAILED -- deployment is live but suspect."
    echo "  *** Roll back:  cd /root/dev/ml_scorer && git checkout <good-sha> && /root/dev/deploy.sh deploy"
    exit 1
  }
  exit 0
fi

echo "usage: $0 {check|deploy}"; exit 2
