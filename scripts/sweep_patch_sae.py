"""L1 sweep for the patch-token SAE (scripts/train_patch_sae.py), to find a sparser SAE.

Reuses the first run's token caches and normalisation (no backbone pass), loads the
training tokens into RAM once, and trains one SAE per L1 coefficient with everything
else unchanged (width, LR, batch, seed, init). Each run is resumable; a finished run
(metrics.json present) is skipped on resubmission.

Per run, <out>/l1_<coeff>/:
  birds525_patch_sae_bundle.pt   weights + metadata (as train_patch_sae's bundle)
  birds525_patch_sae_{best,last}.pt, training_history.json, training_curves.png
  metrics.json, firing_density.png, top_latents/*.png, naming_data.pt
  results/test_results.pt        CPU-only analysis; patches in <source>/results/test_patches.npy

Across runs:
  sweep_summary.json   per model: L1 coeff, epochs, L0, L0 %, mean L1 norm, L2 (MSE), EV,
                       dead %, density histogram + percentiles (validation and test)
  sweep_frontier.png   explained variance vs L0, one point per model

The first run's 100-epoch model is evaluated as a reference ("l1_1e-03_100ep").

Usage:
  python scripts/sweep_patch_sae.py --data /shared/data/birds525 --source_run /shared/sae_patch16 \
      --out /shared/sae_patch16_sweep --l1 1e-3 3e-3 1e-2 3e-2 1e-1
"""

import argparse
import io
import json
import os
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_patch_sae import (ReLUSAE, crop_224, latent_figure, latent_stats, load_into_ram,
                                     load_split, log, plot_density, plot_history, plt, rank_latents,
                                     run_epoch)


def train_one(sae, train_tokens, val_tokens, mean_d, scale_d, run, cfg, base_payload, device):
    """Train (or resume) one SAE; returns its history. Same early stopping as train_patch_sae."""
    best_path = os.path.join(run, "birds525_patch_sae_best.pt")
    last_path = os.path.join(run, "birds525_patch_sae_last.pt")
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    generator = torch.Generator().manual_seed(cfg.seed)
    history, best_loss, ref, stale, start = [], float("inf"), float("inf"), 0, 1
    if os.path.exists(last_path):
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        assert last["config"]["l1_coeff"] == cfg.l1_coeff, f"{run} holds another L1 coefficient"
        sae.load_state_dict(last["state_dict"]); optimizer.load_state_dict(last["optimizer_state"])
        generator.set_state(last["loader_generator_state"])
        history, best_loss, ref = last["history"], last["best_loss"], last["patience_reference"]
        stale, start = last["epochs_without_improvement"], last["epoch"] + 1
        log(f"  resuming at epoch {start}")
        del last

    stopped = stale >= cfg.patience
    for epoch in range(start, cfg.epochs + 1):
        if stopped:
            break
        tr = run_epoch(sae, train_tokens, mean_d, scale_d, cfg, device, optimizer, generator)
        va = run_epoch(sae, val_tokens, mean_d, scale_d, cfg, device)
        history.append({"epoch": epoch, "train": tr, "validation": va})
        pl = dict(base_payload, state_dict={k: v.detach().cpu() for k, v in sae.state_dict().items()},
                  epoch=epoch, validation=va, history=history, config=vars(cfg))
        improved = va["loss"] < best_loss
        if improved:
            best_loss = va["loss"]
            torch.save(pl, best_path + ".part"); os.replace(best_path + ".part", best_path)
        if va["loss"] < ref - cfg.min_delta:
            ref, stale = va["loss"], 0
        else:
            stale += 1
        pl.update({"optimizer_state": optimizer.state_dict(), "loader_generator_state": generator.get_state(),
                   "best_loss": best_loss, "patience_reference": ref, "epochs_without_improvement": stale})
        torch.save(pl, last_path + ".part"); os.replace(last_path + ".part", last_path)
        json.dump(history, open(os.path.join(run, "training_history.json"), "w"), indent=2)
        plot_history(history, os.path.join(run, "training_curves.png"))
        log(f"  epoch {epoch}/{cfg.epochs} | val loss {va['loss']:.5f} | MSE {va['mse']:.5f} | "
            f"L0 {va['l0']:.1f} | EV {va['explained_variance']:.3f} | {tr['seconds'] / 60:.1f} min"
            f"{' | best' if improved else ''} | stale {stale}/{cfg.patience}")
        stopped = stale >= cfg.patience
    if stopped:
        log("  early stopping")
    return history


def evaluate_one(sae, mm, mean_d, scale_d, device, run, ctx, figures=True):
    """Metrics, ranking, figures and CPU-analysis exports for the SAE currently in `sae`."""
    sae.eval()
    scores, density, metrics = {}, {}, {}
    for s in ("validation", "test"):
        scores[s], density[s], metrics[s] = latent_stats(sae, mm[s][0], mean_d, scale_d, device)
    plot_density(density, os.path.join(run, "firing_density.png"))
    scan = {s: rank_latents(scores[s], ctx["cls"][s]) for s in ("validation", "test")}
    if figures:
        fig_dir = os.path.join(run, "top_latents")
        shutil.rmtree(fig_dir, ignore_errors=True); os.makedirs(fig_dir)
        sc, test = scan["test"], ctx["data"]["test"]
        for r, latent in enumerate(sc["order"][:ctx["top_latents"]].tolist()):
            v, i = scores["test"][:, latent].topk(min(ctx["top_images"], len(scores["test"])))
            keep = v > 0
            if keep.any():
                latent_figure(sae, latent, i[keep].tolist(), v[keep].tolist(), mm["test"][0], ctx["crops"],
                              test.targets, ctx["classes"], mean_d, scale_d, ctx["grid"],
                              sc["coherence"][latent].item(), device,
                              os.path.join(fig_dir, f"rank{r + 1:02d}_latent{latent}.png"))
    return scores, density, metrics, scan


def export_one(sae, best, run, scores, density, metrics, scan, history, ctx, mean, scale):
    state = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
    arch = {"input_dim": sae.input_dim, "latent_dim": sae.latent_dim, "encoder_bias": True,
            "decoder_bias": False, "pre_encoder_bias": True}
    bundle = dict(best, state_dict=state, format_version=1, sae_type="ReLU", architecture=arch,
                  normalization="training_mean_and_global_rms", preprocessing=ctx["preprocessing"],
                  grid_size=ctx["grid"], class_names=ctx["classes"],
                  versions={"torch": torch.__version__, "torchvision": __import__("torchvision").__version__})
    torch.save(bundle, os.path.join(run, "birds525_patch_sae_bundle.pt"))
    rel = lambda ds: [os.path.relpath(p, ctx["data_root"]) for p, _ in ds.samples]  # noqa: E731
    val = ctx["data"]["validation"]
    torch.save({"val_scores": scores["validation"], "val_cls": ctx["cls"]["validation"],
                "val_labels": torch.tensor(val.targets), "val_paths": rel(val), "scan": scan["validation"],
                "density": density["validation"], "class_names": ctx["classes"], "data_root": ctx["data_root"],
                "note": "val_scores = max patch activation per image; scan as the notebook's"},
               os.path.join(run, "naming_data.pt"))
    res = os.path.join(run, "results")
    os.makedirs(res, exist_ok=True)
    test = ctx["data"]["test"]
    torch.save({
        "sae_state_dict": state, "architecture": arch, "embedding_mean": mean, "embedding_scale": scale,
        "grid_size": ctx["grid"], "backbone": best["backbone"], "class_names": ctx["classes"],
        "labels": torch.tensor(test.targets), "paths": rel(test), "images_jpeg_224": ctx["jpegs"],
        "scores": scores["test"], "cls": ctx["cls"]["test"], "scan": scan["test"], "density": density["test"],
        "val_density": density["validation"], "metrics": metrics, "history": history,
        "config": best["config"], "best_epoch": best["epoch"],
        "patches_file": ctx["patches_file"],
        "note": ("Test-set results. scores = max patch activation per image x latent. Patch activations: "
                 "relu(((patches - mean) / scale - b_dec) @ W_enc.T + b_enc); patches in patches_file."),
    }, os.path.join(res, "test_results.pt"))


def summary_entry(name, coeff, epochs_trained, best_epoch, metrics):
    keys = ("l0", "l0_pct", "l1", "mse", "explained_variance", "dead", "dead_pct",
            "density_mean", "density_hist", "density_percentiles")
    return {"name": name, "l1_coeff": coeff, "epochs_trained": epochs_trained, "best_epoch": best_epoch,
            **{s: {k: metrics[s][k] for k in keys} for s in ("validation", "test")},
            "notes": {"l0": "active latents per patch", "l1": "mean L1 norm of latents per patch",
                      "mse": "L2 reconstruction error per dimension (normalised tokens)"}}


def plot_frontier(entries, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for e in entries:
        v = e["validation"]
        ref = e["name"].endswith("100ep")
        ax.scatter(v["l0"], v["explained_variance"], s=70, marker="s" if ref else "o",
                   color="gray" if ref else "tab:blue", zorder=3)
        ax.annotate(f"{e['l1_coeff']:g}" + (" (100 ep)" if ref else ""), (v["l0"], v["explained_variance"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=10)
    ax.set_xscale("log")
    ax.axvline(0.05 * 1536, color="tab:red", ls="--", lw=1, label="L0 = 5% of latents")
    ax.set_xlabel("L0: active latents per patch (log scale)"); ax.set_ylabel("Explained variance (validation)")
    ax.set_title("Sparsity vs reconstruction, one point per L1 coefficient"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--source_run", required=True, help="train_patch_sae run: its cache/ and best checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--l1", type=float, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_latents", type=int, default=10)
    p.add_argument("--top_images", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="must match the source run's --limit (smoke tests)")
    args = p.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.source_run, "cache")

    log("===== setup =====")
    ref = torch.load(os.path.join(args.source_run, "birds525_patch_sae_best.pt"), map_location="cpu", weights_only=True)
    cache_sig = json.load(open(os.path.join(cache, "train_cache.json")))
    assert ref["cache_signature"] == cache_sig, "source checkpoint and cache do not match"
    from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights
    patch = ref["config"]["patch_size"]
    preprocess = {16: ViT_B_16_Weights, 32: ViT_B_32_Weights}[patch].IMAGENET1K_V1.transforms()
    data = {s: load_split(args.data, s, preprocess, args.limit) for s in ("validation", "test")}
    mm = {}
    for s in ("train", "validation", "test"):
        sig = json.load(open(os.path.join(cache, f"{s}_cache.json")))
        if s in data:
            assert sig["n"] == len(data[s]), f"{s}: cache has {sig['n']} images, dataset {len(data[s])}"
        mm[s] = (np.load(os.path.join(cache, f"{s}_patches.npy"), mmap_mode="r"),
                 np.load(os.path.join(cache, f"{s}_cls.npy"), mmap_mode="r"))
    D = mm["train"][0].shape[2]
    grid = int(round(mm["train"][0].shape[1] ** 0.5))
    mean, scale = ref["embedding_mean"], ref["embedding_scale"]  # same normalisation as the first run
    mean_d, scale_d = mean.to(device), scale.to(device)
    classes = open(os.path.join(args.data, "classes.txt")).read().splitlines()
    crops = [crop_224(path, preprocess) for path, _ in data["test"].samples]
    jpegs = []
    for c in crops:
        buf = io.BytesIO(); c.save(buf, "JPEG", quality=92); jpegs.append(buf.getvalue())
    ctx = {"data": data, "crops": crops, "jpegs": jpegs, "classes": classes, "grid": grid, "data_root": args.data,
           "cls": {s: torch.from_numpy(np.asarray(mm[s][1], dtype=np.float32)) for s in ("validation", "test")},
           "top_latents": args.top_latents, "top_images": args.top_images,
           "patches_file": os.path.join(args.source_run, "results", "test_patches.npy"),
           "preprocessing": {"weights": "IMAGENET1K_V1", "resize_size": list(preprocess.resize_size),
                             "crop_size": list(preprocess.crop_size), "mean": list(preprocess.mean),
                             "std": list(preprocess.std), "interpolation": preprocess.interpolation.value,
                             "antialias": preprocess.antialias}}
    log(f"grid {grid}x{grid}, dim {D}; L1 values {args.l1}; {args.epochs} epochs, patience {args.patience}")

    summary_path = os.path.join(args.out, "sweep_summary.json")
    entries = {}
    if os.path.exists(summary_path):
        entries = {e["name"]: e for e in json.load(open(summary_path))["models"]}

    def write_summary():
        order = sorted(entries.values(), key=lambda e: (e["l1_coeff"], e["epochs_trained"]))
        json.dump({"source_run": args.source_run, "data": args.data, "latent_dim": ref["latent_dim"],
                   "target": "L0 < 5% of latents", "models": order}, open(summary_path, "w"), indent=2)
        plot_frontier(order, os.path.join(args.out, "sweep_frontier.png"))

    log("===== reference: first run (100 epochs) =====")
    name = f"l1_{ref['config']['l1_coeff']:.0e}_100ep"
    if name not in entries:
        sae = ReLUSAE(ref["input_dim"], ref["latent_dim"] // ref["input_dim"]).to(device)
        sae.load_state_dict(ref["state_dict"])
        _, _, metrics, _ = evaluate_one(sae, mm, mean_d, scale_d, device, args.out, ctx, figures=False)
        os.replace(os.path.join(args.out, "firing_density.png"), os.path.join(args.out, f"firing_density_{name}.png"))
        entries[name] = summary_entry(name, ref["config"]["l1_coeff"], len(ref["history"]), ref["epoch"], metrics)
        write_summary()
    m = entries[name]["validation"]
    log(f"{name}: L0 {m['l0']:.1f} ({m['l0_pct']:.1%}) | EV {m['explained_variance']:.3f} | dead {m['dead_pct']:.1%}")

    log("===== load training tokens =====")
    t0 = time.time()
    train_tokens = load_into_ram(mm["train"][0])
    val_tokens = load_into_ram(mm["validation"][0])
    log(f"train {tuple(train_tokens.shape)} + validation {tuple(val_tokens.shape)} in RAM ({time.time() - t0:.0f}s)")
    base_payload = {"input_dim": D, "latent_dim": ref["latent_dim"], "embedding_mean": mean, "embedding_scale": scale,
                    "backbone": ref["backbone"], "representation": ref["representation"], "cache_signature": cache_sig,
                    "dataset": ref["dataset"], "dataset_revision": ref["dataset_revision"], "dataset_path": args.data}

    for coeff in args.l1:
        name = f"l1_{coeff:.0e}"
        run = os.path.join(args.out, name)
        os.makedirs(run, exist_ok=True)
        if os.path.exists(os.path.join(run, "metrics.json")) and name in entries:
            log(f"===== {name}: done, skipping =====")
            continue
        log(f"===== {name} =====")
        t0 = time.time()
        torch.manual_seed(args.seed)  # same init for every coefficient
        sae = ReLUSAE(D, ref["latent_dim"] // D).to(device)
        cfg = SimpleNamespace(l1_coeff=coeff, lr=args.lr, batch_size=args.batch_size, grad_clip=args.grad_clip,
                              epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, seed=args.seed,
                              expansion=ref["latent_dim"] // D, patch_size=patch)
        history = train_one(sae, train_tokens, val_tokens, mean_d, scale_d, run, cfg, base_payload, device)
        best = torch.load(os.path.join(run, "birds525_patch_sae_best.pt"), map_location="cpu", weights_only=True)
        sae.load_state_dict(best["state_dict"])
        scores, density, metrics, scan = evaluate_one(sae, mm, mean_d, scale_d, device, run, ctx)
        metrics.update(best_epoch=best["epoch"], epochs_run=len(history), l1_coeff=coeff)
        export_one(sae, best, run, scores, density, metrics, scan, history, ctx, mean, scale)
        json.dump(metrics, open(os.path.join(run, "metrics.json"), "w"), indent=2)
        entries[name] = summary_entry(name, coeff, len(history), best["epoch"], metrics)
        write_summary()
        m = metrics["validation"]
        log(f"{name}: best epoch {best['epoch']}/{len(history)} | L0 {m['l0']:.1f} ({m['l0_pct']:.1%}) | "
            f"EV {m['explained_variance']:.3f} | MSE {m['mse']:.5f} | dead {m['dead_pct']:.1%} | "
            f"{(time.time() - t0) / 60:.0f} min")

    log("===== summary =====")
    for e in sorted(entries.values(), key=lambda e: (e["l1_coeff"], e["epochs_trained"])):
        v = e["validation"]
        log(f"  {e['name']:>16}: L0 {v['l0']:7.1f} ({v['l0_pct']:6.2%}) | EV {v['explained_variance']:.3f} | "
            f"dead {v['dead_pct']:6.1%}")
    log("===== all done =====")


if __name__ == "__main__":
    main()
