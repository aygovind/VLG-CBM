"""Results file for another split (e.g. validation), in the format of results/test_results.pt.

train_patch_sae.py / sweep_patch_sae.py export full CPU-analysis results only for the test
set; naming latents on the test set would leak into the final evaluation. This builds the
same file for a run's validation split from what the run already saved: the SAE bundle,
naming_data.pt (validation max-patch scores, CLS features, labels, paths) and the cached
validation patch tokens; it adds the 224 x 224 crops the backbone saw. CPU only.

  python -m scripts.export_split_results --run_dir /shared/sae_patch16_width/w4x_l1_3e-03 \
      --patches /shared/sae_patch16/cache/validation_patches.npy --out .../results/val_results.pt
"""

import argparse
import io
import os

import numpy as np
import torch
from PIL import Image


def crop_224(path, preprocess):
    """The 224 x 224 view the backbone sees, as JPEG bytes."""
    x = preprocess(Image.open(path).convert("RGB")).permute(1, 2, 0).numpy()
    x = np.clip(x * np.array(preprocess.std) + np.array(preprocess.mean), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((x * 255).round().astype(np.uint8)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


@torch.no_grad()
def max_patch_scores(sd, mean, scale, patches, ids, latents):
    """Recompute max-over-patches activations for a few images and latents (sanity check)."""
    x = (torch.from_numpy(np.asarray(patches[ids], dtype=np.float32)) - mean) / scale
    z = torch.relu((x - sd["b_dec"]) @ sd["encoder.weight"][latents].T + sd["encoder.bias"][latents])
    return z.amax(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="folder with birds525_patch_sae_bundle.pt and naming_data.pt")
    p.add_argument("--patches", required=True, help="the split's cached patch tokens (e.g. cache/validation_patches.npy)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    bundle = torch.load(os.path.join(args.run_dir, "birds525_patch_sae_bundle.pt"), map_location="cpu", weights_only=False)
    nd = torch.load(os.path.join(args.run_dir, "naming_data.pt"), map_location="cpu", weights_only=False)
    patches = np.load(args.patches, mmap_mode="r")
    n = len(nd["val_labels"])
    assert patches.shape[0] == n, f"{patches.shape[0]} patch rows vs {n} validation images"

    # --- sanity check: stored scores match the SAE on the cached tokens ---
    sd = bundle["state_dict"]
    ids, lats = [0, n // 2, n - 1], [0, 1, 2]
    rec = max_patch_scores(sd, bundle["embedding_mean"], bundle["embedding_scale"], patches, ids, lats)
    assert torch.allclose(rec, nd["val_scores"][ids][:, lats], atol=1e-3), "scores do not match the cached tokens"

    from torchvision.models import ViT_B_16_Weights
    preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    jpegs = [crop_224(os.path.join(nd["data_root"], rel), preprocess) for rel in nd["val_paths"]]
    out = {
        "sae_state_dict": sd, "architecture": bundle["architecture"],
        "embedding_mean": bundle["embedding_mean"], "embedding_scale": bundle["embedding_scale"],
        "grid_size": bundle["grid_size"], "backbone": bundle["backbone"], "class_names": nd["class_names"],
        "labels": nd["val_labels"], "paths": nd["val_paths"], "images_jpeg_224": jpegs,
        "scores": nd["val_scores"], "cls": nd["val_cls"], "scan": nd["scan"], "density": nd["density"],
        "patches_file": os.path.abspath(args.patches), "split": "validation",
        "note": "Validation-split results in the test_results.pt format, for naming latents without touching test.",
    }
    torch.save(out, args.out)
    print(f"{n} validation images, {out['scores'].shape[1]} latents -> {args.out} "
          f"({os.path.getsize(args.out) / 2**20:.0f} MiB); score check ok")


if __name__ == "__main__":
    main()
