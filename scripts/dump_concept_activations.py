"""Dump per-test-image concept activations, ground truth and prediction to JSON.

Neither repo persists this. VLG-CBM saves concept features only for the two splits
that feed the GLM solve -- `train` and `val`, where `val` is a 10% slice *of train*
(data/concept_dataset.py:221). The held-out birds525 `valid/` folder is the split the
code calls `test`, and its activations are computed inline by test_model /
per_class_accuracy and discarded; only per-class accuracy survives, in metrics.txt.
LF-CBM does cache backbone activations for its `_val` split -- which *is* the held-out
folder -- so that side needs no images at all.

Hence two modes:

  --pipeline lfcbm   Pure matmul on cached activations. CPU, seconds, no images.
                     A = (val_target_features @ W_c.T - proj_mean) / proj_std
  --pipeline vlgcbm  Forward pass of backbone + CBL over the test loader. Needs a GPU
                     and the dataset. ~1 min/model on a 3090.

Both converge on the same output, so the two pipelines stay directly comparable.

Output, per model:
  <out>/<run_name>.json          records + metadata, human-readable
  <out>/<run_name>_acts.npz      full activation matrix, W_g, b_g (float16 acts)

The JSON holds top-k contributions rather than all 2012 activations per image -- the
full matrix as JSON would be ~5.3M floats. The npz carries everything, so any analysis
the JSON does not anticipate is still one np.load away.

Contribution of concept j to class c for image i is W_g[c, j] * A[i, j]: the signed
logit contribution, which is the quantity that answers "which concepts drove this
prediction". Both `pred` and `gt` top-k are recorded, so misclassifications can be read
as a contrast between the two.

Examples:
  python scripts/dump_concept_activations.py --pipeline lfcbm \
      --load_dir /workspace/Label-free-CBM/saved_models/bioclip_birds525__lam0p0007 \
      --lfcbm_root /workspace/Label-free-CBM --out /workspace/concept_dumps

  python scripts/dump_concept_activations.py --pipeline vlgcbm \
      --load_dir saved_models/birds525_bioclip --out /workspace/concept_dumps
"""

import argparse
import json
import os


import numpy as np
import torch


def topk_contrib(acts_row, W_g, cls, k, concepts):
    """Top-k concepts by |signed logit contribution| for one image and one class."""
    contrib = W_g[cls] * acts_row
    # rank by magnitude, but report the signed value: a strongly negative contribution
    # is as informative as a positive one when explaining a wrong prediction
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        {
            "concept": concepts[j],
            "idx": int(j),
            "activation": round(float(acts_row[j]), 4),
            "weight": round(float(W_g[cls, j]), 4),
            "contribution": round(float(contrib[j]), 4),
        }
        for j in order
        if abs(contrib[j]) > 1e-6  # W_g is sparse; most entries are exactly zero
    ]


def lfcbm_activation_path(targs, act_dir):
    """Reproduce LF-CBM's utils.get_save_names for the target (backbone) activations.

    Inlined rather than imported: LF-CBM's utils pulls in data_utils -> clip -> ftfy,
    which is not installed in every environment this runs in (the jupyter pod's base
    interpreter, Colab). The naming is three lines and stable; PM_SUFFIX['avg'] is "".
    """
    d_val = targs.dataset + "_val"
    if targs.backbone.startswith("clip_"):
        # the "backbone_" marker distinguishes this from the CLIP-features cache
        return os.path.join(act_dir, "{}_backbone_{}.pt".format(
            d_val, targs.backbone.replace("/", "")))
    return os.path.join(act_dir, "{}_{}_{}.pt".format(
        d_val, targs.backbone, targs.feature_layer))


def load_lfcbm(load_dir, lfcbm_root, device):
    """Concept activations from cached backbone features -- no images, no GPU needed."""
    with open(os.path.join(load_dir, "args.txt")) as f:
        targs = argparse.Namespace(**json.load(f))

    act_dir = os.path.join(lfcbm_root, getattr(targs, "activation_dir", "saved_activations"))
    val_path = lfcbm_activation_path(targs, act_dir)
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            "cached val activations missing: {}\navailable: {}"
            .format(val_path, sorted(os.listdir(act_dir))[:20]))

    # W_c is saved from a trained proj_layer and still carries requires_grad
    torch.set_grad_enabled(False)
    feats = torch.load(val_path, map_location="cpu").float()
    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location="cpu")
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location="cpu")
    W_c = torch.load(os.path.join(load_dir, "W_c.pt"), map_location="cpu")

    # exactly the transform train_cbm.py:186-199 applies before the final layer:
    # project into concept space FIRST, then standardise with train-split statistics.
    # proj_mean/proj_std are concept-space (n_concepts,), not backbone-space -- doing
    # this in the other order is a shape error here, but would silently produce garbage
    # in any pipeline where the two dims happen to match.
    acts = (feats @ W_c.T - proj_mean) / proj_std

    # ImageFolder sorts class dirs, and so does the label file -- same order as training
    from torchvision.datasets import ImageFolder
    val_root = os.path.join(lfcbm_root, "data", targs.dataset, "val")
    labels = np.asarray(ImageFolder(val_root).targets)

    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = f.read().split("\n")
    with open(os.path.join(lfcbm_root, "data", targs.dataset + ".txt")) as f:
        classes = f.read().split("\n")

    if len(labels) != acts.shape[0]:
        raise ValueError(
            "label count {} != activation rows {}. The cached activations and {} are "
            "out of sync.".format(len(labels), acts.shape[0], val_root))

    return acts.numpy(), labels, concepts, classes, vars(targs)


def load_vlgcbm(load_dir, device, batch_size, num_workers):
    """Concept activations by forward pass -- VLG-CBM's CBL is not a plain matmul."""
    import data.utils as data_utils
    from data.concept_dataset import get_concept_dataloader
    from model.cbm import Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer

    with open(os.path.join(load_dir, "args.txt")) as f:
        targs = argparse.Namespace(**json.load(f))
    with open(os.path.join(load_dir, "concepts.txt")) as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(targs.dataset)

    if targs.backbone.startswith("clip_"):
        backbone = BackboneCLIP(targs.backbone, device=device,
                                use_penultimate=targs.use_clip_penultimate)
    else:
        backbone = Backbone(targs.backbone, targs.feature_layer, device)
    # a finetuned backbone is saved alongside the CBL; without this the activations
    # would come from the pretrained weights and silently disagree with metrics.txt
    ckpt_path = os.path.join(load_dir, "backbone.pt")
    if os.path.exists(ckpt_path):
        backbone.backbone.load_state_dict(torch.load(ckpt_path, map_location=device))

    cbl = ConceptLayer.from_pretrained(load_dir, device)
    norm = NormalizationLayer.from_pretrained(load_dir, device=device)

    # split="test" is the held-out valid/ folder (concept_dataset.py:208)
    loader = get_concept_dataloader(
        targs.dataset, "test", concepts, backbone.preprocess,
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        val_split=None, confidence_threshold=targs.cbl_confidence_threshold,
        crop_to_concept_prob=0.0, label_dir=targs.annotation_dir,
        use_allones=targs.allones_concept, seed=targs.seed,
    )

    acts, labels = [], []
    backbone.eval()
    cbl.eval()
    with torch.no_grad():
        for features, _, y in loader:
            a = norm(cbl(backbone(features.to(device))))
            acts.append(a.detach().cpu())
            labels.append(y)
    acts = torch.cat(acts).numpy()
    labels = torch.cat(labels).numpy()
    return acts, labels, concepts, classes, vars(targs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=["lfcbm", "vlgcbm"], required=True)
    p.add_argument("--load_dir", required=True, help="a trained CBM run directory")
    p.add_argument("--out", default="concept_dumps")
    p.add_argument("--lfcbm_root", default=None,
                   help="LF-CBM checkout; required for --pipeline lfcbm")
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_npz", action="store_true",
                   help="skip the full activation matrix (JSON only)")
    args = p.parse_args()

    if args.pipeline == "lfcbm":
        if not args.lfcbm_root:
            p.error("--lfcbm_root is required for --pipeline lfcbm")
        acts, labels, concepts, classes, targs = load_lfcbm(
            args.load_dir, os.path.abspath(args.lfcbm_root), args.device)
    else:
        acts, labels, concepts, classes, targs = load_vlgcbm(
            args.load_dir, args.device, args.batch_size, args.num_workers)

    W_g = torch.load(os.path.join(args.load_dir, "W_g.pt"), map_location="cpu").float().numpy()
    b_g = torch.load(os.path.join(args.load_dir, "b_g.pt"), map_location="cpu").float().numpy()

    if acts.shape[1] != W_g.shape[1]:
        raise ValueError(
            "concept dim mismatch: activations {} vs W_g {}. The run dir and the "
            "cached activations are probably from different training runs."
            .format(acts.shape[1], W_g.shape[1]))
    if len(concepts) != W_g.shape[1]:
        raise ValueError("concepts.txt has {} entries but W_g has {} columns"
                         .format(len(concepts), W_g.shape[1]))

    logits = acts @ W_g.T + b_g
    preds = logits.argmax(axis=1)
    acc = float((preds == labels).mean())

    run_name = os.path.basename(os.path.normpath(args.load_dir))
    print("{}: {} images, {} concepts, {} classes | test acc {:.4f}".format(
        run_name, acts.shape[0], acts.shape[1], len(classes), acc))
    # a recomputed accuracy that disagrees with metrics.txt means the activations do not
    # correspond to this model -- worth knowing before any analysis is built on them
    mpath = os.path.join(args.load_dir, "metrics.txt")
    if os.path.exists(mpath):
        with open(mpath) as f:
            m = json.load(f)
        ref = m.get("metrics", {}).get("test_accuracy") or m.get("metrics", {}).get("acc_val")
        if ref is not None:
            ref = float(ref)
            ref = ref / 100.0 if ref > 1.0 else ref
            flag = "OK" if abs(ref - acc) < 0.02 else "MISMATCH"
            print("  metrics.txt reports {:.4f} -> {}".format(ref, flag))

    nz = np.abs(W_g) > 1e-5
    records = []
    for i in range(acts.shape[0]):
        gt, pr = int(labels[i]), int(preds[i])
        rec = {
            "index": i,
            "gt": gt,
            "gt_class": classes[gt],
            "pred": pr,
            "pred_class": classes[pr],
            "correct": gt == pr,
            "pred_logit": round(float(logits[i, pr]), 4),
            "gt_logit": round(float(logits[i, gt]), 4),
            "top_concepts_pred": topk_contrib(acts[i], W_g, pr, args.topk, concepts),
        }
        if gt != pr:
            # on a miss, the gt-class concepts are the other half of the story
            rec["top_concepts_gt"] = topk_contrib(acts[i], W_g, gt, args.topk, concepts)
        records.append(rec)

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "run_name": run_name,
        "pipeline": args.pipeline,
        "load_dir": os.path.abspath(args.load_dir),
        "split": "test (held-out birds525 valid/ folder)",
        "n_images": int(acts.shape[0]),
        "n_concepts": int(acts.shape[1]),
        "n_classes": len(classes),
        "test_accuracy": acc,
        "topk": args.topk,
        "sparsity": {
            "nonzero_weights": int(nz.sum()),
            "total_weights": int(W_g.size),
            "concepts_with_outgoing_weight": int(nz.any(axis=0).sum()),
            "mean_concepts_per_class": float(nz.sum(axis=1).mean()),
        },
        "train_args": targs,
        "concepts": concepts,
        "classes": classes,
        "records": records,
    }
    jpath = os.path.join(args.out, run_name + ".json")
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=1)
    print("  wrote {} ({:.1f} MB)".format(jpath, os.path.getsize(jpath) / 1e6))

    if not args.no_npz:
        npath = os.path.join(args.out, run_name + "_acts.npz")
        np.savez_compressed(
            npath,
            activations=acts.astype(np.float16),  # 4 sig figs is plenty for analysis
            labels=labels.astype(np.int16),
            preds=preds.astype(np.int16),
            W_g=W_g.astype(np.float32),
            b_g=b_g.astype(np.float32),
            concepts=np.array(concepts, dtype=object),
            classes=np.array(classes, dtype=object),
        )
        print("  wrote {} ({:.1f} MB)".format(npath, os.path.getsize(npath) / 1e6))


if __name__ == "__main__":
    main()
