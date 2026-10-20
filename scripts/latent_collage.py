"""Random-latent collage for a patch SAE: an unbiased qualitative check.

The top_latents/ figures show the most coherent latents, which flatters any SAE. This
picks n_latents latents at random (seeded) among those that fire on at least n_images
test images, and shows each one's top n_images test images with its patch activation
map, one latent per row. Each row is labelled with the latent id, the mean pairwise CLS
cosine similarity of the images shown, and the latent's firing density.

Needs only a results file from train_patch_sae.py / sweep_patch_sae.py
(results/test_results.pt) and the test patches it points to, so it runs on a CPU, locally
or in any pod that mounts the volume. No repository imports.

  from scripts.latent_collage import random_latent_collage
  random_latent_collage("l1_3e-03/results/test_results.pt", "collage.png", seed=0)

  python scripts/latent_collage.py --results run_a/results/test_results.pt run_b/... \
      --out_dir collages --seed 0
"""

import argparse
import io
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def load_results(path):
    """test_results.pt plus a memory map of the test patches it refers to."""
    r = torch.load(path, map_location="cpu", weights_only=False)
    patches = r["patches_file"]
    if not os.path.isabs(patches):  # the first run stores it next to the results file
        patches = os.path.join(os.path.dirname(path), patches)
    r["patches"] = np.load(patches, mmap_mode="r")
    assert r["patches"].shape[0] == len(r["labels"]), "patches and results cover different images"
    return r


@torch.no_grad()
def latent_maps(r, image_ids, latent):
    """[n, grid, grid] activations of one latent on the given test images."""
    sd = r["sae_state_dict"]
    x = torch.from_numpy(np.asarray(r["patches"][image_ids], dtype=np.float32))
    x = (x - r["embedding_mean"]) / r["embedding_scale"]
    z = torch.relu((x - sd["b_dec"]) @ sd["encoder.weight"][latent] + sd["encoder.bias"][latent])
    g = r["grid_size"]
    return z.reshape(len(image_ids), g, g).numpy()


def pick_latents(r, n_latents, n_images, seed):
    """Seeded random latents among those positive on >= n_images test images."""
    alive = torch.where((r["scores"] > 0).sum(0) >= n_images)[0].numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(alive, size=min(n_latents, len(alive)), replace=False)
    return sorted(chosen.tolist()), len(alive)


def random_latent_collage(results, out_path, n_latents=5, n_images=5, seed=0, title=None, latents=None):
    """Collage of n_latents random latents (rows) x their top n_images test images.
    results: a path or an already loaded results dict. latents: override the random pick.
    Returns the latent ids shown."""
    r = load_results(results) if isinstance(results, str) else results
    if latents is None:
        latents, n_alive = pick_latents(r, n_latents, n_images, seed)
    else:
        n_alive = int(((r["scores"] > 0).sum(0) >= n_images).sum())
    if not latents:
        print(f"no latent fires on >= {n_images} test images; nothing to draw")
        return []
    feats = F.normalize(r["cls"].float(), dim=-1)
    density = r["density"]
    classes = r["class_names"]

    fig, axes = plt.subplots(len(latents), n_images, figsize=(3.3 * n_images, 3.6 * len(latents)), squeeze=False)
    for row, latent in zip(axes, latents):
        v, ids = r["scores"][:, latent].topk(n_images)
        ids, v = ids[v > 0].tolist(), v[v > 0].tolist()
        sim = feats[ids] @ feats[ids].T
        iu = torch.triu_indices(len(ids), len(ids), offset=1)
        cos = sim[iu[0], iu[1]].mean().item() if len(ids) > 1 else float("nan")
        maps = latent_maps(r, ids, latent)
        vmax = max(maps.max(), 1e-8)  # one colour scale per row (latents differ in scale)
        for k, ax in enumerate(row):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k >= len(ids):
                continue
            ax.imshow(Image.open(io.BytesIO(r["images_jpeg_224"][ids[k]])))
            ax.imshow(maps[k], extent=(0, 224, 224, 0), cmap="magma", alpha=.5,
                      interpolation="nearest", vmin=0, vmax=vmax)
            ax.set_title(f"{classes[int(r['labels'][ids[k]])].title()}\nact {v[k]:.2f}", fontsize=10)
        row[0].set_ylabel(f"latent {latent}\ncos sim {cos:.3f}\nfires on {density[latent]:.1%}",
                          fontsize=12, rotation=0, ha="right", va="center", labelpad=12)
    if title is None:  # run folder name: <run>/results/test_results.pt
        title = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(results)))) \
            if isinstance(results, str) else ""
    model = title
    fig.suptitle(f"{model}: {len(latents)} random latents (seed {seed}, from {n_alive} firing on "
                 f">= {n_images} test images), top {n_images} test images each", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return latents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True, help="results/test_results.pt files")
    p.add_argument("--names", nargs="+", default=None, help="one per results file (default: run folder name)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_latents", type=int, default=5)
    p.add_argument("--n_images", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    names = args.names or [os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(f)))) for f in args.results]
    for path, name in zip(args.results, names):
        out = os.path.join(args.out_dir, f"random_latents_{name}_seed{args.seed}.png")
        shown = random_latent_collage(path, out, args.n_latents, args.n_images, args.seed, title=name)
        print(f"{name}: latents {shown} -> {out}", flush=True)


if __name__ == "__main__":
    main()
