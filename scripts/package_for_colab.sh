#!/bin/bash
# Tar up exactly what the Colab notebook needs and nothing else: val-split images,
# val-split annotations, the requested trained runs, and (only if bioclip v1 is among
# them) its checkpoint. Everything else the notebook needs -- concept_files, the repo
# code -- comes from `git clone` and does not need to travel.
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
mkdir -p "$OUT/data/birds525" "$OUT/annotations" "$OUT/saved_models" "$OUT/models/bioclip"

echo "birds525 val images..."
cp -r datasets/birds525/val "$OUT/data/birds525/val"

echo "birds525 val annotations..."
cp -r annotations/birds525_val "$OUT/annotations/birds525_val"

need_bioclip1=false
for m in "${MODELS[@]}"; do
  src="saved_models/birds525_${m}"
  test -d "$src" || { echo "FATAL: no saved_models/birds525_${m}"; exit 1; }
  echo "run: $m..."
  cp -r "$src" "$OUT/saved_models/birds525_${m}"
  [ "$m" == "bioclip" ] && need_bioclip1=true
done

if [ "$need_bioclip1" = true ]; then
  echo "bioclip v1 checkpoint (bioclip2 and the others pull from public hubs, no staging needed)..."
  cp models/bioclip/open_clip_pytorch_model.bin "$OUT/models/bioclip/"
fi

tar czf colab_package.tar.gz "$OUT"
rm -rf "$OUT"
echo "wrote colab_package.tar.gz: $(du -sh colab_package.tar.gz | cut -f1)"
