#!/bin/bash
# Build birds525mini: the first N classes of birds525, symlinked rather than copied.
#
# Annotating and training on it end to end takes minutes, which is the point -- it
# exercises exactly the same code path as the real birds525 jobs (Grounding DINO
# prompts, ConceptDataset indexing, BioCLIP backbone, CBL, GLM-SAGA), so a bug shows
# up before the multi-hour jobs are submitted rather than during them.
#
# Usage: bash scripts/make_smoke_dataset.sh [n_classes]
set -eo pipefail

N=${1:-5}
ROOT=${DATASET_FOLDER:-datasets}
SRC="$ROOT/birds525"
DST="$ROOT/birds525mini"

test -d "$SRC/train" || { echo "FATAL: $SRC/train not found"; exit 1; }

rm -rf "$DST"
for split in train val; do
  mkdir -p "$DST/$split"
  # ImageFolder assigns labels by sorted directory name, and the classes file is in
  # that same order, so taking the first N of each keeps the two aligned.
  ls "$SRC/$split" | sort | head -n "$N" | while read -r cls; do
    ln -s "$(cd "$SRC/$split/$cls" && pwd)" "$DST/$split/$cls"
  done
done

echo "birds525mini: $N classes"
for split in train val; do
  echo "  $split: $(find -L "$DST/$split" -name '*.jpg' | wc -l) images"
done
