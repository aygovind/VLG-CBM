"""SAGE-CBM: patch-SAE latents as the concept layer, VLG-CBM's sparse final layer on top.

  image -> ViT-B/16 patch tokens -> frozen SAE encoder -> max over patches (one score per latent)
        -> z-score with training-set mean/std -> FinalLayer trained with GLM-SAGA

Everything from the concept scores on is VLG-CBM's code path (model.cbm.FinalLayer,
model.cbm.train_sparse_final, the same standardisation as data.concept_dataset.
get_final_layer_dataset), so SAGE-CBM and a VLG-CBM baseline differ only in the concept
layer. No concept supervision, no Grounding DINO.

Stages:
  scores   per split, max-over-patches activation of every latent, from the cached patch
           tokens of train_patch_sae.py (cache/<split>_patches.npy) -> <out>/scores_<split>.pt
  train    for each concept subset (--subset all | a CSV with a 'latent' column) and each
           lambda: z-score (train stats), GLM-SAGA, test accuracy, per-class accuracy, sparsity
           -> <out>/<subset>_lam<lambda>/{final.pt, train_concept_features_mean/std.pt, metrics.json}
           and <out>/summary.csv

  python -m scripts.sage_cbm --run_dir /shared/sae_patch16_width/w4x_l1_3e-03 \
      --cache /shared/sae_patch16/cache --data /shared/data/birds525 --out /shared/sage_cbm/w4x \
      --subset all kept=/shared/sage_cbm/concepts_kept.csv --lam 0.0007 0.001 0.0015 0.002 --device cpu
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "validation", "test")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- scores

@torch.no_grad()
def concept_scores(bundle, patches, chunk=256):
    """[N, L] max-over-patches SAE activation per image (>= 0), from fp16 cached tokens."""
    sd = bundle["state_dict"]
    W, b, b_dec = sd["encoder.weight"].float(), sd["encoder.bias"].float(), sd["b_dec"].float()
    mean, scale = bundle["embedding_mean"].float(), bundle["embedding_scale"].float()
    out = torch.empty(patches.shape[0], W.shape[0])
    t0 = time.time()
    for s in range(0, patches.shape[0], chunk):
        x = (torch.from_numpy(np.asarray(patches[s:s + chunk], dtype=np.float32)) - mean) / scale
        out[s:s + chunk] = torch.relu((x - b_dec) @ W.T + b).amax(1)
        if (s // chunk) % 50 == 0:
            done = min(s + chunk, patches.shape[0])
            log(f"    {done:,}/{patches.shape[0]:,} images ({done / (time.time() - t0):.0f} img/s)")
    return out


def labels_for(data_root, split):
    """ImageFolder targets: the order the token cache was written in (train_patch_sae.py)."""
    from torchvision.datasets import ImageFolder
    return torch.tensor(ImageFolder(os.path.join(data_root, split)).targets)


# ---------------------------------------------------------------- final layer

def standardize(train, *others):
    """VLG-CBM's normalisation: per-concept z-score with training-set statistics."""
    mean, std = train.mean(0), train.std(0).clamp_min(1e-6)              # floor: latents silent on train
    return mean, std, [(x - mean) / std for x in (train, *others)]


def fit_final(z_train, y_train, z_val, y_val, n_classes, lam, args):
    """GLM-SAGA exactly as VLG-CBM's train_cbm.py: zero-init FinalLayer, alpha 0.99, eps 1, k 1."""
    from glm_saga.elasticnet import IndexedTensorDataset
    from model.cbm import FinalLayer, train_sparse_final
    torch.manual_seed(args.seed)
    train_loader = DataLoader(IndexedTensorDataset(z_train.to(args.device), y_train), batch_size=args.saga_batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(z_val.to(args.device), y_val), batch_size=args.saga_batch_size, shuffle=False)
    layer = FinalLayer(z_train.shape[1], n_classes, device=args.device)
    out = train_sparse_final(layer, train_loader, val_loader, args.saga_n_iters, lam,
                             step_size=args.saga_step_size, device=args.device)
    p = out["path"][0]
    return p["weight"].cpu(), p["bias"].cpu(), p


def evaluate(W, b, z, y, n_classes):
    pred = (z @ W.T + b).argmax(1)
    per_class = torch.zeros(n_classes)
    for c in range(n_classes):
        m = y == c
        per_class[c] = (pred[m] == c).float().mean() if m.any() else float("nan")
    return (pred == y).float().mean().item(), per_class


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="SAE run folder with birds525_patch_sae_bundle.pt")
    p.add_argument("--cache", required=True, help="token cache folder with <split>_patches.npy")
    p.add_argument("--data", required=True, help="cleaned Birds-525 (labels and class names)")
    p.add_argument("--out", required=True)
    p.add_argument("--subset", nargs="+", default=["all"], help="'all' and/or name=path.csv (column 'latent')")
    p.add_argument("--lam", type=float, nargs="+", default=[0.0007], help="GLM-SAGA lambda(s); 0.0007 = VLG-CBM config")
    p.add_argument("--saga_n_iters", type=int, default=1000)
    p.add_argument("--saga_step_size", type=float, default=0.1)
    p.add_argument("--saga_batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="first N images per split (smoke tests)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    log("===== 1. concept scores =====")
    bundle = torch.load(os.path.join(args.run_dir, "birds525_patch_sae_bundle.pt"), map_location="cpu", weights_only=False)
    classes = open(os.path.join(args.data, "classes.txt")).read().splitlines()
    assert classes == bundle["class_names"], "class list differs from the SAE run's"
    S, Y = {}, {}
    for split in SPLITS:
        path = os.path.join(args.out, f"scores_{split}.pt")
        Y[split] = labels_for(args.data, split)[:args.limit]
        if os.path.exists(path):
            S[split] = torch.load(path)
            log(f"{split}: loaded {tuple(S[split].shape)}")
        else:
            patches = np.load(os.path.join(args.cache, f"{split}_patches.npy"), mmap_mode="r")[:args.limit]
            assert patches.shape[0] == len(Y[split]), f"{split}: {patches.shape[0]} cached vs {len(Y[split])} images"
            log(f"{split}: {patches.shape[0]:,} images")
            S[split] = concept_scores(bundle, patches)
            torch.save(S[split], path)
        assert (S[split] >= 0).all() and S[split].shape[0] == len(Y[split])
    log(f"scores: train {tuple(S['train'].shape)}; latents never on in train: {(S['train'].amax(0) == 0).sum().item()}")

    log("===== 2. final layer (VLG-CBM GLM-SAGA) =====")
    rows = []
    for spec in args.subset:
        name, idx = ("all", torch.arange(S["train"].shape[1])) if spec == "all" else \
            (spec.split("=")[0], torch.tensor(sorted(pd.read_csv(spec.split("=")[1])["latent"].astype(int).tolist())))
        mean, std, (zt, zv, zs) = standardize(S["train"][:, idx], S["validation"][:, idx], S["test"][:, idx])
        for lam in args.lam:
            run = os.path.join(args.out, f"{name}_lam{lam:g}")
            if os.path.exists(os.path.join(run, "metrics.json")):
                rows.append(json.load(open(os.path.join(run, "metrics.json")))["summary"]); log(f"{run}: done, skipping")
                continue
            os.makedirs(run, exist_ok=True)
            log(f"{name}: {len(idx)} concepts, lambda {lam:g}")
            t0 = time.time()
            W, b, path = fit_final(zt, Y["train"], zv, Y["validation"], len(classes), lam, args)
            test_acc, per_class = evaluate(W, b, zs, Y["test"], len(classes))
            val_acc, _ = evaluate(W, b, zv, Y["validation"], len(classes))
            nz = W.abs() > 1e-5
            summary = {"subset": name, "n_concepts": len(idx), "lam": lam, "test_acc": test_acc, "val_acc": val_acc,
                       "nonzero_weights": int(nz.sum()), "per_class_mean": nz.sum(1).float().mean().item(),
                       "concepts_used": int(nz.any(0).sum()), "minutes": (time.time() - t0) / 60}
            torch.save({"weight": W, "bias": b}, os.path.join(run, "final.pt"))
            torch.save(mean, os.path.join(run, "train_concept_features_mean.pt"))
            torch.save(std, os.path.join(run, "train_concept_features_std.pt"))
            torch.save(idx, os.path.join(run, "latents.pt"))
            json.dump({"summary": summary, "per_class_accuracies": dict(zip(classes, per_class.tolist())),
                       "saga": {k: float(path[k]) for k in ("lam", "lr", "alpha", "time")},
                       "settings": {k: getattr(args, k) for k in ("saga_n_iters", "saga_step_size",
                                                                  "saga_batch_size", "seed")}},
                      open(os.path.join(run, "metrics.json"), "w"), indent=2)
            rows.append(summary)
            log(f"  test acc {test_acc:.2%} | val {val_acc:.2%} | {summary['per_class_mean']:.1f} concepts/class | "
                f"{summary['concepts_used']} concepts used | {summary['minutes']:.1f} min")
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    log("===== summary =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
