"""Generate eval_val.pt for every VLG-CBM CBM, plus a self-contained copy to download.

vlgcbm_analysis.evaluate() already computes everything needed -- predictions, labels,
logits and the normalized concept activations the final layer consumes, for all 2,625
held-out test images -- and caches it as <run>/eval_val.pt. But that cache is written
lazily, only when a notebook happens to call evaluate() on a model, which is why most
runs do not have one. This script calls it on every run so the set is complete.

It deliberately goes through vlgcbm_analysis rather than reimplementing the forward
pass: the notebooks read eval_val.pt, so generating it with the same code path is the
only way to guarantee they see identical numbers.

Two outputs per run:
  <run_dir>/eval_val.pt     the notebook cache, unchanged format
  <out>/<name>.pt           self-contained bundle: the same tensors plus concept names,
                            class names, W_g and b_g, so it can be analysed on a laptop
                            with nothing but torch -- no repo, no PVC, no model weights

Bundle contents (N = 2625 test images, C = concepts, K = 525 classes):
  preds         [N]     int64    predicted class index
  labels        [N]     int64    ground-truth class index
  correct       [N]     bool
  logits        [N,K]   float32
  concept_acts  [N,C]   float32  normalized CBL output == the final layer's input
  W_g           [K,C]   float32  final layer weights
  b_g           [K]     float32
  concepts      list[str], length C
  classes       list[str], length K
  meta          dict: run_dir, accuracy, metrics.txt accuracy, train args

Per-concept contribution to class k for image i is concept_acts[i] * W_g[k] -- the
signed logit contribution. logits[i] == concept_acts[i] @ W_g.T + b_g, which the script
asserts, so any analysis built on the bundle is built on the tensors the model used.

Usage:
  python scripts/generate_eval.py --save_dir saved_models --out /shared/vlgcbm_eval
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlgcbm_analysis as va  # noqa: E402


def bundle_name(load_dir, save_dir):
    """birds525_bioclip for saved_models/birds525_bioclip/birds525_cbm_<ts>.

    Uses the config-level parent, which is the name the notebooks use (run_dir("bioclip")),
    rather than the timestamped leaf, which says nothing about the backbone.
    """
    rel = os.path.relpath(load_dir, save_dir).split(os.sep)
    return rel[0]


def reference_accuracy(load_dir):
    """Test accuracy train_cbm.py recorded, for a sanity check against our recompute."""
    path = os.path.join(load_dir, "metrics.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    acc = m.get("metrics", {}).get("test_accuracy")
    if acc is None:
        return None
    acc = float(acc)
    return acc / 100.0 if acc > 1.0 else acc


def process(load_dir, save_dir, out_dir, device, batch_size, num_workers, force):
    name = bundle_name(load_dir, save_dir)
    started = time.time()
    run = va.load_run(load_dir, device=device)
    # cache=True: reuse a valid existing eval_val.pt rather than re-running the backbone,
    # and write one where it is missing. --force recomputes and overwrites.
    res = va.evaluate(run, split="val", batch_size=batch_size,
                      num_workers=num_workers, cache=not force)
    if force:
        # evaluate(cache=False) neither reads nor writes the cache; write it explicitly
        # so --force still leaves the notebook cache refreshed
        torch.save({"preds": res.preds, "labels": res.labels,
                    "concept_acts": res.concept_acts, "logits": res.logits,
                    "n_concepts": len(run.concepts)},
                   os.path.join(load_dir, "eval_val.pt"))

    final = run.model.final
    W_g = final.weight.detach().float().cpu()
    b_g = final.bias.detach().float().cpu()

    acts = res.concept_acts.float()
    logits = res.logits.float()
    # The bundle is only useful if concept_acts @ W_g reproduces what the model decided.
    # A mismatch would mean the stored activations are not the final layer's true input
    # -- e.g. pre-normalization -- and every contribution computed from them is wrong.
    recon = acts @ W_g.T + b_g
    max_err = (recon - logits).abs().max().item()
    if max_err > 1e-2:
        raise AssertionError(
            "concept_acts @ W_g.T + b_g does not reproduce the stored logits "
            "(max abs err {:.4g}); the activations are not the final layer's input"
            .format(max_err))

    acc = res.accuracy
    ref = reference_accuracy(load_dir)

    bundle = {
        "preds": res.preds.long(),
        "labels": res.labels.long(),
        "correct": res.correct,
        "logits": logits,
        "concept_acts": acts,
        "W_g": W_g,
        "b_g": b_g,
        "concepts": list(run.concepts),
        "classes": list(run.classes),
        "meta": {
            "name": name,
            "run_dir": os.path.abspath(load_dir),
            "split": "birds525_val (held-out test set, 5 images/class)",
            "accuracy": acc,
            "metrics_txt_accuracy": ref,
            "logit_reconstruction_max_err": max_err,
            "n_images": int(len(res.labels)),
            "n_concepts": len(run.concepts),
            "n_classes": len(run.classes),
            "nonzero_weights": int((W_g.abs() > 1e-5).sum()),
            "train_args": run.train_args,
        },
    }
    path = os.path.join(out_dir, name + ".pt")
    torch.save(bundle, path)

    agree = ""
    if ref is not None:
        agree = " | metrics.txt {:.4f} -> {}".format(
            ref, "OK" if abs(ref - acc) < 0.01 else "MISMATCH")
    print("  {}: acc {:.4f}{} | {} concepts | recon err {:.1e} | {:.1f} MB | {:.0f}s".format(
        name, acc, agree, len(run.concepts), max_err,
        os.path.getsize(path) / 1e6, time.time() - started), flush=True)

    del run, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"name": name, "run_dir": load_dir, "accuracy": acc, "metrics_txt": ref,
            "n_concepts": bundle["meta"]["n_concepts"], "status": "ok"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save_dir", default="saved_models")
    p.add_argument("--out", required=True, help="directory for the downloadable bundles")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force", action="store_true",
                   help="recompute even where a valid eval_val.pt exists")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these bundle names, e.g. birds525_bioclip")
    args = p.parse_args()

    runs = va.find_runs(args.save_dir, newest_only=True)
    if args.only:
        runs = [r for r in runs if bundle_name(r, args.save_dir) in args.only]
    if not runs:
        # fail loudly: an empty loop that exits 0 is indistinguishable from success
        sys.exit("no complete CBM runs found under {} (need {})".format(
            args.save_dir, ", ".join(va.ARTIFACTS)))

    print("found {} CBM run(s):".format(len(runs)))
    for r in runs:
        print("  " + r)
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for r in runs:
        print("\n=== {} ===".format(r), flush=True)
        try:
            summary.append(process(r, args.save_dir, args.out, args.device,
                                   args.batch_size, args.num_workers, args.force))
        except Exception as exc:  # one bad run must not cost the others
            traceback.print_exc()
            summary.append({"name": bundle_name(r, args.save_dir), "run_dir": r,
                            "status": "failed", "error": repr(exc)})

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    failed = [s for s in summary if s["status"] != "ok"]
    print("\n{} ok, {} failed".format(len(summary) - len(failed), len(failed)))
    for s in failed:
        print("  FAILED {}: {}".format(s["name"], s["error"]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
