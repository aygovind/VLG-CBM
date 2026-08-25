#!/bin/bash
# Tar up exactly what the Colab notebook needs and nothing else: val-split images,
# val-split annotations, and the trained run artifacts. Backbone weights do NOT travel --
# every backbone resolves from a public source at load time. Everything else the notebook
# needs -- concept_files, the repo code -- comes from `git clone`.
#
# Run on the pod, from the VLG-CBM repo root:
#   bash scripts/package_for_colab.sh bioclip vit_in21k > /tmp/models.txt   # which models
#   kubectl cp <pod>:/workspace/VLG-CBM/colab_package.tar.gz colab_package.tar.gz
# Then upload colab_package.tar.gz to Google Drive and point the notebook's DATA_ROOT
# at wherever you extract it.
set -eo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  echo "Usage: $0 <model> [model...]   e.g. $0 bioclip vit_in21k dino clip_vitb16 rn50 bioclip2"
  exit 1
fi

OUT=colab_package
rm -rf "$OUT"
mkdir -p "$OUT/data/birds525" "$OUT/annotations" "$OUT/saved_models"

echo "birds525 val images..."
cp -r datasets/birds525/val "$OUT/data/birds525/val"

echo "birds525 val annotations..."
cp -r annotations/birds525_val "$OUT/annotations/birds525_val"

# Explicit allowlist, not `cp -r`. get_final_layer_dataset caches the standardised
# concept feature matrix next to the run for resuming training -- train_concept_features.pt
# alone is ~550MB (76,172 x 1,911 float32), none of which vlgcbm_analysis.load_run() or
# evaluate() reads. Copying a run directory verbatim was measured at ~630MB; this list
# gets the same run down to a few MB. eval_val.pt is the one optional extra worth keeping:
# it is evaluate()'s own prediction cache, so including it lets Colab skip a fresh
# inference pass over the whole eval set.
KEEP_FILES=(cbl.pt final.pt concepts.txt concept_counts.txt args.txt
            train_concept_features_mean.pt train_concept_features_std.pt eval_val.pt)

for m in "${MODELS[@]}"; do
  src="saved_models/birds525_${m}"
  test -d "$src" || { echo "FATAL: no saved_models/birds525_${m}"; exit 1; }
  run=$(ls -td "$src"/*/ 2>/dev/null | head -1)
  test -n "$run" || { echo "FATAL: no run subdirectory under $src"; exit 1; }
  dst="$OUT/saved_models/birds525_${m}/$(basename "$run")"
  mkdir -p "$dst"
  echo "run: $m ($(basename "$run"))..."
  for f in "${KEEP_FILES[@]}"; do
    [ -f "$run/$f" ] && cp "$run/$f" "$dst/$f"
  done
done

# No backbone weights travel in this package. Every backbone resolves from a public
# source at load time -- bioclip and bioclip2 from the HF hub, vit_in21k/dino via timm,
# clip_vitb16 via open_clip, rn50 via torchvision. Shipping BioCLIP v1's checkpoint here
# added 570MB, ~85% of the tarball, for weights the target machine can fetch itself.

tar czf colab_package.tar.gz "$OUT"
rm -rf "$OUT"
echo "wrote colab_package.tar.gz: $(du -sh colab_package.tar.gz | cut -f1)"
