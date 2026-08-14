"""Run a list of VLG-CBM runs in one process, skipping ones already on disk.

train_cbm.py writes to a fresh timestamped directory on every invocation, so re-running
a config silently retrains it rather than resuming. On NRP a multi-run suite will be
evicted at least once, so --skip_existing checks each config's save_dir for a run that
got as far as writing metrics.txt (train_cbm.py writes it last, after the test pass) and
skips those. A run killed mid-way leaves no metrics.txt and is redone from scratch.

Runs are subprocesses rather than imported calls: train_cbm.py and train_standard.py
both reconfigure global state (loguru sinks, sys.stdout, the random seed), and a crash
in one run should not take the rest of the suite with it.

Usage:
    python run_suite.py --suite backbone_comparison --skip_existing
    python run_suite.py --suite backbone_comparison --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Each entry is (kind, config-or-flags). "cbm" runs train_cbm.py against a config file;
# "standard" runs train_standard.py, the black-box linear probe on the same backbone,
# which takes flags rather than a config.
#
# Ordering is deliberate: every standard run first, then every CBM run. The standard runs
# are ~30min each and touch the same backbone-loading path, so a broken backbone surfaces
# within the first hour instead of four hours into a CBM run. bioclip leads each group
# because it is the one already validated end to end.
SUITES = {
    "backbone_comparison": [
        ("standard", "bioclip"),
        ("standard", "vit_in21k"),
        ("standard", "dino"),
        ("standard", "clip_vitb16"),
        ("standard", "rn50"),
        ("cbm", "bioclip"),
        ("cbm", "vit_in21k"),
        ("cbm", "dino"),
        ("cbm", "clip_vitb16"),
        ("cbm", "rn50"),
    ],
}

# train_standard.py takes flags, so the backbone spec has to be repeated here. Kept in
# sync with configs/birds525_<name>.json by check_consistency() below.
STANDARD_BACKBONES = {
    "bioclip": ("bioclip", "visual.ln_post"),
    "vit_in21k": ("vit_in21k", "out"),
    "dino": ("dino_vitb16", "out"),
    "clip_vitb16": ("clip_ViT-B/16", "unused"),
    "rn50": ("resnet50", "layer4"),
}


def config_path(name):
    return os.path.join("configs", f"birds525_{name}.json")


def check_consistency():
    """Fail loudly if a standard run would use a different backbone than its CBM twin."""
    for name, (backbone, layer) in STANDARD_BACKBONES.items():
        with open(config_path(name)) as f:
            cfg = json.load(f)
        if cfg["backbone"] != backbone:
            raise ValueError(
                f"{name}: config says backbone {cfg['backbone']!r}, "
                f"STANDARD_BACKBONES says {backbone!r}")
        if not backbone.startswith("clip_") and cfg["feature_layer"] != layer:
            raise ValueError(
                f"{name}: config says feature_layer {cfg['feature_layer']!r}, "
                f"STANDARD_BACKBONES says {layer!r}")


def save_dir_for(kind, name):
    with open(config_path(name)) as f:
        cfg = json.load(f)
    base = cfg["save_dir"]
    return base if kind == "cbm" else base + "_standard"


def is_complete(kind, name):
    """A run counts as done once some subdirectory holds a metrics.txt."""
    root = save_dir_for(kind, name)
    if not os.path.isdir(root):
        return False
    for entry in os.listdir(root):
        if os.path.exists(os.path.join(root, entry, "metrics.txt")):
            return True
    return False


def stale_against_config(kind, name):
    """Hyperparameters where a finished run disagrees with the config it would run under.

    --skip_existing keys on "a metrics.txt exists", so a run finished under different
    settings is skipped and silently stays in the results at those settings. In a suite
    whose whole point is that only the backbone varies, that is the failure worth
    shouting about: it does not crash, it just quietly produces a comparison that is not
    controlled. train_cbm.py dumps the settings it used to args.txt, so they can be
    compared directly.
    """
    if kind != "cbm":
        return {}
    root = save_dir_for(kind, name)
    with open(config_path(name)) as f:
        cfg = json.load(f)
    for entry in sorted(os.listdir(root)):
        args_path = os.path.join(root, entry, "args.txt")
        if not (os.path.exists(args_path) and os.path.exists(os.path.join(root, entry, "metrics.txt"))):
            continue
        with open(args_path) as f:
            used = json.load(f)
        return {k: (used[k], v) for k, v in cfg.items()
                if k in used and used[k] != v and k not in ("save_dir", "load_dir", "annotation_dir")}
    return {}


def command_for(kind, name, annotation_dir):
    if kind == "cbm":
        return [sys.executable, "train_cbm.py", "--config", config_path(name),
                "--annotation_dir", annotation_dir]

    backbone, layer = STANDARD_BACKBONES[name]
    with open(config_path(name)) as f:
        cfg = json.load(f)
    return [sys.executable, "train_standard.py",
            "--dataset", cfg["dataset"],
            "--backbone", backbone,
            "--feature_layer", layer,
            "--device", cfg["device"],
            "--batch_size", "256",
            # match the CBM runs' final layer so the two are comparable
            "--lam", str(cfg["saga_lam"]),
            "--n_iters", str(cfg["saga_n_iters"]),
            "--save_dir", save_dir_for(kind, name)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="backbone_comparison", choices=sorted(SUITES))
    parser.add_argument("--annotation_dir", type=str, default="annotations")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip runs whose save_dir already holds a finished run")
    parser.add_argument("--dry_run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    check_consistency()
    runs = SUITES[args.suite]
    failures = []
    stale_skips = []

    for i, (kind, name) in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] {kind}:{name}"
        cmd = command_for(kind, name, args.annotation_dir)

        if args.dry_run:
            print(f"{tag}\n    -> {save_dir_for(kind, name)}\n    $ {' '.join(cmd)}")
            continue

        if args.skip_existing and is_complete(kind, name):
            stale = stale_against_config(kind, name)
            if stale:
                diffs = ", ".join(f"{k}: on disk {was!r} != config {now!r}" for k, (was, now) in stale.items())
                print(f"{tag} already complete -- skipping, but IT DOES NOT MATCH THE CONFIG: {diffs}",
                      flush=True)
                print(f"{' ' * len(tag)}   the suite will compare it against runs trained differently; "
                      f"move {save_dir_for(kind, name)} aside to redo it", flush=True)
                stale_skips.append(f"{kind}:{name}")
            else:
                print(f"{tag} already complete -- skipping", flush=True)
            continue

        print(f"\n{'=' * 70}\n{tag}\n$ {' '.join(cmd)}\n{'=' * 70}", flush=True)
        started = time.time()
        result = subprocess.run(cmd)
        elapsed = (time.time() - started) / 60

        if result.returncode != 0:
            # Keep going: one broken backbone should not cost the suite the runs behind it,
            # and --skip_existing means a resubmit retries only what failed.
            print(f"{tag} FAILED with exit {result.returncode} after {elapsed:.0f} min", flush=True)
            failures.append(f"{kind}:{name}")
        else:
            print(f"{tag} finished in {elapsed:.0f} min", flush=True)

    if stale_skips:
        print(f"\nWARNING: {len(stale_skips)} run(s) skipped despite not matching the current "
              f"config: {', '.join(stale_skips)}")
    if failures:
        print(f"\n{len(failures)} run(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("\nsuite complete")


if __name__ == "__main__":
    main()
