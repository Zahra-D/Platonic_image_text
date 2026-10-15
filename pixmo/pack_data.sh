#!/bin/bash
# Pack the prepared PixMo data ($PIXMO_DATA, without the per-shard intermediates) for another cluster.
#   PIXMO_DATA=~/platonic_pixmo_data bash pixmo/pack_data.sh ~/platonic_pixmo_data.tar
set -euo pipefail
OUT=${1:?usage: pack_data.sh OUTPUT.tar}
DATA=${PIXMO_DATA:-$HOME/platonic_pixmo_data}
tar -cf "$OUT" -C "$(dirname "$DATA")" --exclude=parts "$(basename "$DATA")"
ls -lh "$OUT"
