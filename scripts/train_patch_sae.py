"""Patch-token SAE on Birds-525: the Kaggle notebook sae-concept-set-generation-patch-trial,
sections 1-3, as a resumable job, with ViT-B/16 (14 x 14 = 196 spatial tokens).

Same model and objective as the notebook: torchvision ViT-B/16 (IMAGENET1K_V1), final
LayerNormed spatial tokens (CLS excluded), normalised by the training mean and global RMS,
into a 768 -> 1536 -> 768 ReLU SAE (pre-encoder bias, unit-norm decoder columns),
loss = MSE + L1_COEFF * sum(z), Adam, gradient clipping, early stopping on the
validation objective. Data: the cleaned Birds-525 copy (scripts/prepare_birds525.py).

Stages (each resumes or is skipped when its output exists):
  1. cache      fp16 patch tokens + CLS for train / validation / test (.npy); a progress
                file lets an evicted pod continue where it stopped
  2. normalise  training mean and global RMS (float64 sums)
  3. train      up to --epochs, patience --patience; last + best checkpoints every epoch,
                training_history.json and training_curves.png
  4. analyse    MSE, L0, firing density, dead / rare latents (validation and test)
  5. rank       max patch activation per image; latents ranked by the mean CLS cosine
                similarity of their top-12 images (validation for naming, test for review)
  6. figures    top_latents/: one PNG per top-ranked latent, its top test images with
                patch activation maps
  7. export     birds525_patch_sae_bundle.pt (as the notebook's export), naming_data.pt,
                and results/ for CPU-only analysis: test_results.pt + test_patches.npy

Difference from the notebook: the training tokens (25 GB fp16) are held in RAM and
shuffled per epoch with one permutation, drawn in blocks; the notebook's per-token reads
from a memory-mapped file would be bound by random disk reads at this size.

Usage:
  python scripts/train_patch_sae.py --data /shared/data/birds525 --run_dir /shared/sae_patch16
  python scripts/train_patch_sae.py ... --limit 64 --epochs 2 --random_backbone --device cpu   # smoke test
"""

import argparse
import io
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ("train", "validation", "test")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- backbone + data

def load_vit(patch_size, device, random_backbone=False):
    """Pretrained torchvision ViT-B/{patch_size} with the head removed (forward -> CLS)."""
    from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32
    constructor, weights = {16: (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
                            32: (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1)}[patch_size]
    model = constructor(weights=None if random_backbone else weights)
    model.heads = nn.Identity()
    model = model.to(device).eval().requires_grad_(False)
    return model, weights.transforms()


@torch.inference_mode()
def encode_tokens(vit, images):
    """Final encoder tokens [B, 1 + patches, 768] (CLS first), as in the notebook."""
    images = images.to(next(vit.parameters()).device, non_blocking=True)
    patches = vit._process_input(images)
    cls = vit.class_token.expand(images.shape[0], -1, -1)
    return vit.encoder(torch.cat([cls, patches], dim=1))


def load_split(data_root, split, preprocess, limit=None):
    from torchvision.datasets import ImageFolder
    ds = ImageFolder(os.path.join(data_root, split), transform=preprocess)
    classes = open(os.path.join(data_root, "classes.txt")).read().splitlines()
    assert ds.classes == classes, "ImageFolder classes differ from classes.txt"
    if limit:  # smoke tests: a spread of classes, not just the first one
        keep = np.linspace(0, len(ds.samples) - 1, min(limit, len(ds.samples))).astype(int)
        ds.samples = [ds.samples[i] for i in keep]
        ds.targets = [s[1] for s in ds.samples]
        ds.imgs = ds.samples
    return ds


# ---------------------------------------------------------------- 1. cache

def cache_split(vit, ds, path_prefix, num_patches, dim, batch_size, workers, signature):
    """fp16 [N, P, D] patch tokens and [N, D] CLS, written in order; resumes from a
    progress file. Returns read-only memmaps."""
    patches_path, cls_path = path_prefix + "_patches.npy", path_prefix + "_cls.npy"
    meta_path, prog_path = path_prefix + "_cache.json", path_prefix + "_progress.json"
    n = len(ds)
    if os.path.exists(meta_path):
        if json.load(open(meta_path)) != signature:
            raise SystemExit(f"FATAL: {meta_path} was made with other settings; use a new --run_dir")
        log(f"{os.path.basename(path_prefix)}: cache complete, {n:,} images")
        return np.load(patches_path, mmap_mode="r"), np.load(cls_path, mmap_mode="r")

    done = 0
    if os.path.exists(prog_path) and json.load(open(prog_path))["signature"] == signature:
        done = json.load(open(prog_path))["done"]
    else:  # allocate both files at full size; data is then written in place
        for p, shape in ((patches_path, (n, num_patches, dim)), (cls_path, (n, dim))):
            m = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=shape)
            del m
    offsets = {p: np.load(p, mmap_mode="r").offset for p in (patches_path, cls_path)}
    log(f"{os.path.basename(path_prefix)}: caching {n - done:,} of {n:,} images"
        + (f" (resuming at {done:,})" if done else ""))

    sub = torch.utils.data.Subset(ds, range(done, n))
    loader = torch.utils.data.DataLoader(sub, batch_size=batch_size, num_workers=workers,
                                         pin_memory=torch.cuda.is_available())
    started, last_report = time.time(), time.time()
    with open(patches_path, "r+b") as fp, open(cls_path, "r+b") as fc:
        for images, _ in loader:
            tokens = encode_tokens(vit, images).float().cpu().numpy().astype(np.float16)
            b = len(images)
            fp.seek(offsets[patches_path] + done * num_patches * dim * 2)
            fp.write(np.ascontiguousarray(tokens[:, 1:]).tobytes())
            fc.seek(offsets[cls_path] + done * dim * 2)
            fc.write(np.ascontiguousarray(tokens[:, 0]).tobytes())
            done += b
            if time.time() - last_report > 60 or done == n:
                fp.flush(); fc.flush(); os.fsync(fp.fileno()); os.fsync(fc.fileno())
                json.dump({"signature": signature, "done": done}, open(prog_path, "w"))
                rate = (done - (n - len(sub))) / (time.time() - started)
                log(f"  {done:,}/{n:,} images, {rate:.0f} img/s, ~{(n - done) / max(rate, 1e-9) / 60:.1f} min left")
                last_report = time.time()
    assert done == n
    json.dump(signature, open(meta_path, "w"))
    os.remove(prog_path)
    log(f"{os.path.basename(path_prefix)}: cached in {(time.time() - started) / 60:.1f} min, "
        f"{os.path.getsize(patches_path) / 2**30:.2f} GiB")
    return np.load(patches_path, mmap_mode="r"), np.load(cls_path, mmap_mode="r")


def load_into_ram(memmap, chunk_images=2048):
    """[N, P, D] fp16 memmap -> [N*P, D] fp16 CPU tensor, read sequentially in chunks."""
    n, p, d = memmap.shape
    out = torch.empty(n * p, d, dtype=torch.float16)
    for s in range(0, n, chunk_images):
        e = min(n, s + chunk_images)
        out[s * p:e * p] = torch.from_numpy(np.array(memmap[s:e]).reshape(-1, d))
    return out


# ---------------------------------------------------------------- SAE (as in the notebook)

class ReLUSAE(nn.Module):
    """Encode embeddings into sparse ReLU latents and reconstruct them."""

    def __init__(self, input_dim=768, expansion=2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        with torch.no_grad():
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def encode(self, x):
        return torch.relu(self.encoder(x - self.b_dec))

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z) + self.b_dec, z


# ---------------------------------------------------------------- 3. train

def run_epoch(sae, data, mean, scale, args, device, optimizer=None, generator=None):
    """One pass over [T, D] fp16 tokens. Training if optimizer is given (shuffled by
    generator). Metrics are per token, as in the notebook."""
    training = optimizer is not None
    sae.train(training)
    T = len(data)
    order = torch.randperm(T, generator=generator) if training else None
    block = args.batch_size * 64  # gather 64 batches at a time, then slice on the GPU
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    sum_x = torch.zeros(sae.input_dim, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros_like(sum_x)
    started, last_report = time.time(), time.time()
    with torch.set_grad_enabled(training):
        for bs in range(0, T, block):
            idx = order[bs:bs + block] if training else slice(bs, bs + block)
            xb = data[idx].to(device, non_blocking=True).float()
            xb = (xb - mean) / scale
            for s in range(0, len(xb), args.batch_size):
                x = xb[s:s + args.batch_size]
                rec, z = sae(x)
                mse = F.mse_loss(rec, x)
                loss = mse + args.l1_coeff * z.sum(dim=-1).mean()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
                    optimizer.step()
                    sae.normalize_decoder()
                l0 = (z.detach() > 0).sum(dim=-1).float().mean()
                totals += torch.stack([loss.detach(), mse.detach(), l0]).double() * len(x)
                if not training:
                    sum_x += x.double().sum(0)
                    sum_x2 += x.double().square().sum(0)
            if not torch.isfinite(totals).all():
                raise RuntimeError("Non-finite loss detected.")
            if training and time.time() - last_report > 120:
                done = min(T, bs + block)
                log(f"    {done / T:.0%} of epoch, running loss {totals[0].item() / done:.5f}")
                last_report = time.time()
    loss, mse, l0 = (totals / T).tolist()
    secs = time.time() - started
    out = {"loss": loss, "mse": mse, "l0": l0, "seconds": secs, "tokens_per_second": T / secs}
    if not training:
        var = (sum_x2 / T - (sum_x / T).square()).mean().clamp_min(1e-12).item()
        out["explained_variance"] = 1 - mse / var
    return out


def plot_history(history, path):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for key, ax, title in zip(["loss", "mse", "l0"], axes[:3],
                              ["Objective: MSE + L1 penalty", "Reconstruction MSE", "Active latents per patch (L0)"]):
        for split in ("train", "validation"):
            ax.plot(epochs, [r[split][key] for r in history], "o-", ms=3, label=split)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=.25)
    axes[3].plot(epochs, [r["validation"]["explained_variance"] for r in history], "o-", ms=3)
    axes[3].set_title("Validation explained variance"); axes[3].set_xlabel("Epoch"); axes[3].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4-5. analyse, rank

@torch.inference_mode()
def latent_stats(sae, patches_mm, mean, scale, device, chunk_images=256):
    """Per image max activation over patches [N, L], plus token-level metrics."""
    n, p, d = patches_mm.shape
    firing = torch.zeros(sae.latent_dim, dtype=torch.float64, device=device)
    sq_err = l1 = active = 0.0
    maxes = []
    for s in range(0, n, chunk_images):
        x = torch.from_numpy(np.array(patches_mm[s:s + chunk_images])).to(device).float()
        x = (x - mean) / scale
        rec, z = sae(x)                               # [b, P, L]
        sq_err += (rec - x).square().sum().item()
        l1 += z.sum().item()
        active += (z > 0).sum().item()
        firing += (z > 0).sum(dim=(0, 1)).double()
        maxes.append(z.amax(dim=1).cpu())
    tokens = n * p
    density = (firing / tokens).cpu().numpy()
    metrics = {"mse": sq_err / (tokens * d), "l1": l1 / tokens, "l0": active / tokens,
               "density_mean": float(density.mean()), "density_std": float(density.std()),
               "density_min": float(density.min()), "density_max": float(density.max()),
               "dead": int((density == 0).sum()), "firing_lt_1pct": int((density < 0.01).sum()),
               "firing_lt_5pct": int((density < 0.05).sum()), "firing_lt_10pct": int((density < 0.10).sum())}
    return torch.cat(maxes), density, metrics


def plot_density(densities, path):
    edges = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    labels = ["Dead", "(0–1%)", "[1–5%)", "[5–10%)", "[10–25%)", "[25–50%)", "[50–100%]"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(densities)
    for k, (name, d) in enumerate(densities.items()):
        counts = np.r_[(d == 0).sum(), np.histogram(d[d > 0], bins=edges)[0]]
        ax.bar(np.arange(len(labels)) + (k - (len(densities) - 1) / 2) * w, 100 * counts / len(d), w, label=name)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Latent firing rate (% of patches)"); ax.set_ylabel("% of latents"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def rank_latents(scores, cls_feats, top_n=12):
    """As the notebook: top_n images per latent by max patch activation; latents with
    top_n positive images, ranked by the mean pairwise cosine similarity of their CLS
    features (raw, before SAE normalisation)."""
    feats = F.normalize(cls_feats.float(), dim=-1)
    values, indices = scores.topk(min(top_n, len(scores)), dim=0)
    coherence = torch.full((scores.shape[1],), float("nan"))
    for latent in range(scores.shape[1]):
        ids = indices[:, latent][values[:, latent] > 0]
        if len(ids) < 2:
            continue
        sim = feats[ids] @ feats[ids].T
        iu = torch.triu_indices(len(ids), len(ids), offset=1)
        coherence[latent] = sim[iu[0], iu[1]].mean()
    eligible = torch.where(torch.isfinite(coherence) & (values[-1] > 0))[0]
    order = eligible[coherence[eligible].argsort(descending=True)]
    return {"coherence": coherence, "order": order, "indices": indices, "values": values}


# ---------------------------------------------------------------- 6. figures

def crop_224(path, preprocess):
    """The 224 x 224 view the backbone sees, as a uint8 PIL image."""
    from PIL import Image
    x = preprocess(Image.open(path).convert("RGB")).permute(1, 2, 0).numpy()
    x = np.clip(x * np.array(preprocess.std) + np.array(preprocess.mean), 0, 1)
    return Image.fromarray((x * 255).round().astype(np.uint8))


@torch.inference_mode()
def latent_figure(sae, latent, ids, values, patches_mm, crops, labels, classes, mean, scale,
                  grid, coherence, device, path):
    """Top test images for one latent with its patch activation map (shared colour scale)."""
    n = len(ids)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
    vmax = max(max(values), 1e-8)
    for ax, i, v in zip(axes.flat, ids, values):
        x = torch.from_numpy(np.asarray(patches_mm[i], dtype=np.float32)).to(device)
        amap = sae.encode((x - mean) / scale)[:, latent].cpu().reshape(grid, grid).numpy()
        ax.imshow(crops[i])
        im = ax.imshow(amap, extent=(0, 224, 224, 0), cmap="magma", alpha=.5,
                       interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"#{i} {classes[labels[i]].title()}\nmax act {v:.2f}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=.02, pad=.01)
    fig.suptitle(f"Latent {latent} | top-image CLS cosine similarity {coherence:.3f}", fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="cleaned Birds-525 (scripts/prepare_birds525.py)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--patch_size", type=int, default=16, choices=[16, 32])
    p.add_argument("--expansion", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l1_coeff", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_latents", type=int, default=10, help="latent figures to save")
    p.add_argument("--top_images", type=int, default=10, help="images per latent figure")
    p.add_argument("--image_batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="images per split, smoke tests only")
    p.add_argument("--random_backbone", action="store_true", help="skip the weight download, smoke tests only")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    run = args.run_dir
    cache_dir = os.path.join(run, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    classes = open(os.path.join(args.data, "classes.txt")).read().splitlines()

    log("===== setup =====")
    vit, preprocess = load_vit(args.patch_size, device, args.random_backbone)
    grid = vit.image_size // vit.patch_size
    P, D = grid * grid, vit.hidden_dim
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 224, 224, device=device)
        torch.testing.assert_close(encode_tokens(vit, probe)[:, 0], vit(probe))
    backbone_id = f"torchvision/vit_b_{args.patch_size}/IMAGENET1K_V1" + ("/RANDOM" if args.random_backbone else "")
    log(f"backbone {backbone_id}: grid {grid}x{grid} = {P} tokens of {D}; CLS check ok")
    data = {s: load_split(args.data, s, preprocess, args.limit) for s in SPLITS}
    log("images: " + ", ".join(f"{s} {len(d):,}" for s, d in data.items()) + f"; {len(classes)} classes")

    log("===== 1. cache =====")
    mm = {}
    for s in SPLITS:
        sig = {"n": len(data[s]), "tokens": P, "dim": D, "backbone": backbone_id,
               "data_revision": manifest["revision"], "data_created": manifest["created"],
               "representation": "final_normalized_spatial", "preprocess": repr(preprocess), "limit": args.limit}
        mm[s] = cache_split(vit, data[s], os.path.join(cache_dir, s), P, D, args.image_batch, args.workers, sig)
    del vit
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_path = os.path.join(run, "birds525_patch_sae_best.pt")
    last_path = os.path.join(run, "birds525_patch_sae_last.pt")
    hist_path = os.path.join(run, "training_history.json")
    cache_sig = json.load(open(os.path.join(cache_dir, "train_cache.json")))

    log("===== 2. load train tokens + normalisation =====")
    t0 = time.time()
    train_tokens = load_into_ram(mm["train"][0])
    val_tokens = load_into_ram(mm["validation"][0])
    log(f"train {tuple(train_tokens.shape)} + validation {tuple(val_tokens.shape)} in RAM "
        f"({(train_tokens.numel() + val_tokens.numel()) * 2 / 2**30:.1f} GiB, {time.time() - t0:.0f}s)")
    if os.path.exists(last_path):
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        mean, scale = saved["embedding_mean"], saved["embedding_scale"]
        log("normalisation from the last checkpoint")
    else:
        tot = torch.zeros(D, dtype=torch.float64, device=device)
        sq = torch.zeros_like(tot)
        for s in range(0, len(train_tokens), 1 << 17):  # 131k tokens: 0.8 GB per float64 buffer
            x = train_tokens[s:s + (1 << 17)].to(device).double()
            tot += x.sum(0); sq += x.square().sum(0)
        m64 = tot / len(train_tokens)
        mean = m64.float().cpu()
        scale = (sq / len(train_tokens) - m64.square()).mean().clamp_min(1e-16).sqrt().float().cpu()
    log(f"embedding mean norm {mean.norm():.3f}, global RMS scale {scale.item():.4f}")
    mean_d, scale_d = mean.to(device), scale.to(device)

    log("===== 3. train =====")
    sae = ReLUSAE(D, args.expansion).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    config = {k: getattr(args, k) for k in ("batch_size", "epochs", "patience", "min_delta", "lr",
                                            "l1_coeff", "grad_clip", "seed", "expansion", "patch_size")}
    history, best_loss, ref, stale, start = [], float("inf"), float("inf"), 0, 1
    if os.path.exists(last_path):
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        assert last["cache_signature"] == cache_sig, "checkpoint was trained on another cache"
        assert last["config"]["l1_coeff"] == args.l1_coeff, "use a new --run_dir when changing the objective"
        sae.load_state_dict(last["state_dict"]); optimizer.load_state_dict(last["optimizer_state"])
        generator.set_state(last["loader_generator_state"])
        history, best_loss, ref = last["history"], last["best_loss"], last["patience_reference"]
        stale, start = last["epochs_without_improvement"], last["epoch"] + 1
        log(f"resuming at epoch {start} (best val loss {best_loss:.5f}, {stale} epochs without improvement)")
        del last
    log(f"{len(train_tokens):,} training tokens/epoch = {len(train_tokens) // args.batch_size:,} steps; "
        f"up to {args.epochs} epochs, patience {args.patience}")

    def payload(epoch, metrics):
        return {"state_dict": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
                "input_dim": sae.input_dim, "latent_dim": sae.latent_dim,
                "embedding_mean": mean, "embedding_scale": scale,
                "backbone": backbone_id, "representation": "final_normalized_spatial",
                "cache_signature": cache_sig, "dataset": manifest["source"],
                "dataset_revision": manifest["revision"], "dataset_path": args.data,
                "epoch": epoch, "validation": metrics, "history": history, "config": config}

    stopped_early = stale >= args.patience
    for epoch in range(start, args.epochs + 1):
        if stopped_early:
            break
        tr = run_epoch(sae, train_tokens, mean_d, scale_d, args, device, optimizer, generator)
        va = run_epoch(sae, val_tokens, mean_d, scale_d, args, device)
        history.append({"epoch": epoch, "train": tr, "validation": va})
        pl = payload(epoch, va)
        improved = va["loss"] < best_loss
        if improved:
            best_loss = va["loss"]
            torch.save(pl, best_path + ".part"); os.replace(best_path + ".part", best_path)
        if va["loss"] < ref - args.min_delta:
            ref, stale = va["loss"], 0
        else:
            stale += 1
        pl.update({"optimizer_state": optimizer.state_dict(), "loader_generator_state": generator.get_state(),
                   "best_loss": best_loss, "patience_reference": ref, "epochs_without_improvement": stale})
        torch.save(pl, last_path + ".part"); os.replace(last_path + ".part", last_path)
        json.dump(history, open(hist_path, "w"), indent=2)
        plot_history(history, os.path.join(run, "training_curves.png"))
        log(f"epoch {epoch}/{args.epochs} | train loss {tr['loss']:.5f} | val loss {va['loss']:.5f} "
            f"| val MSE {va['mse']:.5f} | val L0 {va['l0']:.1f} | EV {va['explained_variance']:.3f} "
            f"| {tr['seconds'] / 60:.1f} min, {tr['tokens_per_second']:,.0f} tok/s"
            f"{' | best' if improved else ''} | stale {stale}/{args.patience}")
        stopped_early = stale >= args.patience
    if stopped_early:
        log("early stopping: validation objective plateaued")
    del train_tokens, val_tokens

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    sae.load_state_dict(best["state_dict"]); sae.eval()
    log(f"using best epoch {best['epoch']} (val loss {best['validation']['loss']:.5f})")

    log("===== 4. analyse =====")
    scores, density, metrics = {}, {}, {}
    for s in ("validation", "test"):
        scores[s], density[s], metrics[s] = latent_stats(sae, mm[s][0], mean_d, scale_d, device)
        m = metrics[s]
        log(f"{s}: MSE {m['mse']:.5f} | L0 {m['l0']:.1f}/{sae.latent_dim} | density mean {m['density_mean']:.2%} "
            f"| dead {m['dead']} | <1% {m['firing_lt_1pct']} | <5% {m['firing_lt_5pct']} | <10% {m['firing_lt_10pct']}")
    plot_density(density, os.path.join(run, "firing_density.png"))
    metrics["best_epoch"] = best["epoch"]
    metrics["epochs_run"] = len(history)
    json.dump(metrics, open(os.path.join(run, "metrics.json"), "w"), indent=2)

    log("===== 5. rank =====")
    cls = {s: torch.from_numpy(np.asarray(mm[s][1], dtype=np.float32)) for s in ("validation", "test")}
    scan = {s: rank_latents(scores[s], cls[s]) for s in ("validation", "test")}
    for s in scan:
        log(f"{s}: {len(scan[s]['order'])}/{sae.latent_dim} latents ranked; top 10: {scan[s]['order'][:10].tolist()}")

    log("===== 6. figures =====")
    test = data["test"]
    t0 = time.time()
    crops = [crop_224(path, preprocess) for path, _ in test.samples]
    log(f"{len(crops):,} test crops in {time.time() - t0:.0f}s")
    fig_dir = os.path.join(run, "top_latents")
    shutil.rmtree(fig_dir, ignore_errors=True); os.makedirs(fig_dir)
    sc = scan["test"]
    for r, latent in enumerate(sc["order"][:args.top_latents].tolist()):
        v, i = scores["test"][:, latent].topk(args.top_images)
        keep = v > 0
        latent_figure(sae, latent, i[keep].tolist(), v[keep].tolist(), mm["test"][0], crops, test.targets,
                      classes, mean_d, scale_d, grid, sc["coherence"][latent].item(), device,
                      os.path.join(fig_dir, f"rank{r + 1:02d}_latent{latent}.png"))
    log(f"{min(args.top_latents, len(sc['order']))} figures in {fig_dir}")

    log("===== 7. export =====")
    bundle = dict(best)
    bundle.update({
        "format_version": 1, "sae_type": "ReLU",
        "architecture": {"input_dim": sae.input_dim, "latent_dim": sae.latent_dim, "encoder_bias": True,
                         "decoder_bias": False, "pre_encoder_bias": True},
        "normalization": "training_mean_and_global_rms",
        "preprocessing": {"weights": "IMAGENET1K_V1", "resize_size": list(preprocess.resize_size),
                          "crop_size": list(preprocess.crop_size), "mean": list(preprocess.mean),
                          "std": list(preprocess.std), "interpolation": preprocess.interpolation.value,
                          "antialias": preprocess.antialias},
        "grid_size": grid, "class_names": classes,
        "versions": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__},
    })
    torch.save(bundle, os.path.join(run, "birds525_patch_sae_bundle.pt"))

    rel = lambda ds: [os.path.relpath(path, args.data) for path, _ in ds.samples]  # noqa: E731
    torch.save({"val_scores": scores["validation"], "val_cls": cls["validation"],
                "val_labels": torch.tensor(data["validation"].targets), "val_paths": rel(data["validation"]),
                "scan": scan["validation"], "density": density["validation"], "class_names": classes,
                "data_root": args.data, "note": "val_scores = max patch activation per image; scan as the notebook's"},
               os.path.join(run, "naming_data.pt"))

    res_dir = os.path.join(run, "results")
    os.makedirs(res_dir, exist_ok=True)
    jpegs = []
    for c in crops:
        buf = io.BytesIO(); c.save(buf, "JPEG", quality=92); jpegs.append(buf.getvalue())
    torch.save({
        "sae_state_dict": bundle["state_dict"], "architecture": bundle["architecture"],
        "embedding_mean": mean, "embedding_scale": scale, "grid_size": grid, "backbone": backbone_id,
        "class_names": classes, "labels": torch.tensor(test.targets), "paths": rel(test),
        "images_jpeg_224": jpegs, "scores": scores["test"], "cls": cls["test"],
        "scan": scan["test"], "density": density["test"], "val_density": density["validation"],
        "metrics": metrics, "history": history, "config": config, "best_epoch": best["epoch"],
        "patches_file": "test_patches.npy",
        "note": ("Test-set results. scores = max patch activation per image x latent. "
                 "Patch activations: relu(((patches - mean) / scale - b_dec) @ W_enc.T + b_enc), "
                 "patches from test_patches.npy [N, grid*grid, 768] fp16."),
    }, os.path.join(res_dir, "test_results.pt"))
    shutil.copyfile(os.path.join(cache_dir, "test_patches.npy"), os.path.join(res_dir, "test_patches.npy"))
    for f in sorted(os.listdir(res_dir)):
        log(f"  results/{f}: {os.path.getsize(os.path.join(res_dir, f)) / 2**20:.0f} MiB")
    log("===== all done =====")


if __name__ == "__main__":
    main()
