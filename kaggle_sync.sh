#!/usr/bin/env bash
# Kaggle <-> local sync for the DRL trading notebook (Option A: API push/pull).
#
#   ./kaggle_sync.sh data         create the public dataset (first time only)
#   ./kaggle_sync.sh data-update  push new data as a new dataset version
#   ./kaggle_sync.sh push         upload notebooks/ and start a GPU run
#   ./kaggle_sync.sh status       poll the run
#   ./kaggle_sync.sh pull         download run outputs -> notebooks/out/ and embed
#                                 them back into notebooks/drl_trading_pipeline.ipynb
#   ./kaggle_sync.sh log          tail the pulled run_log.txt
#
# Requires ~/.kaggle/kaggle.json (chmod 600). You edit the notebook locally;
# the run happens on Kaggle's GPU.
set -euo pipefail

# New-style Kaggle access token (KGAT_...) must be in the env for write ops
# (create/version/push); the kaggle.json key alone only authenticates reads.
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "$HOME/.kaggle/access_token" ]]; then
  export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="niko980/drl-trading-pipeline"
DATA_DIR="$REPO/data/raw"
NB_DIR="$REPO/notebooks"
OUT_DIR="$NB_DIR/out"

case "${1:-help}" in
  data)        kaggle datasets create  -p "$DATA_DIR" -u ;;
  data-update) kaggle datasets version -p "$DATA_DIR" -m "update $(date -u +%FT%TZ)" ;;
  push)        kaggle kernels push   -p "$NB_DIR"; echo "Pushed -> track with: $0 status" ;;
  status)      kaggle kernels status "$KERNEL" ;;
  pull)        mkdir -p "$OUT_DIR"; kaggle kernels output "$KERNEL" -p "$OUT_DIR"; echo "Outputs -> $OUT_DIR"
               # Kaggle's API returns working-dir artifacts + a flat log, but NOT
               # the executed notebook. Reattach the run's real outputs (per-cell
               # stdout + the equity-curves figure) so the repo .ipynb shows them.
               python "$REPO/embed_kaggle_outputs.py" ;;
  log)         tail -n 60 "$OUT_DIR/run_log.txt" 2>/dev/null || echo "No run_log.txt yet — run '$0 pull' after a run finishes." ;;
  *)           grep -E '^#( |!)' "$0" | sed 's/^#[ !]\{0,1\}//' ;;
esac
