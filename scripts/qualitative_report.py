"""Qualitative comparison of VLG-CBM runs as image files -- the notebook, without the notebook.

For every run it writes the three prediction collages (random, confidently correct,
confidently wrong). Then, for one shared set of test images, it writes each run's
explanation -- the image with Grounding DINO's boxes next to the top concept
contributions -- stacked one run above the other, so the same image can be read
across concept sets.

Everything goes through vlgcbm_analysis, the same code path as evaluate_vlgcbm.ipynb,
so the figures match what the notebook would show.

Images are chosen so the sheet is worth reading: first the ones the runs disagree on
(one right, one wrong), then random ones up to --n. Pass --indices to pick them.

Output, under --out:
  class_accuracy.txt         per-class accuracy of every run, side by side
  <name>/collage_{random,confident_correct,confident_wrong}.png
  compare_<idx>.png          one image, every run's explanation stacked
  comparison.png             all of the above in one tall sheet

Usage:
  python scripts/qualitative_report.py --out figures/sae_vs_control \
      --run sae=saved_models/birds525sae_v2_vit_in21k:annotations_sae \
      --run control=saved_models/birds525_v2_vit_in21k:annotations
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display on the pod; must precede any pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlgcbm_analysis as va  # noqa: E402


def resolve_run_dir(path):
    """Accept a run directory or its config-level parent (newest complete run wins)."""
    def complete(p):
        return all(os.path.exists(os.path.join(p, f)) for f in va.ARTIFACTS)

    if complete(path):
        return path
    nested = sorted(os.path.join(path, s) for s in os.listdir(path) if complete(os.path.join(path, s)))
    if not nested:
        sys.exit("no complete run under {} (need {})".format(path, ", ".join(va.ARTIFACTS)))
    return nested[-1]


def pick_indices(results, n, seed):
    """Images the runs disagree on first, then random ones, n in total."""
    correct = torch.stack([r.correct for r in results.values()])
    disagree = (correct.any(dim=0) & ~correct.all(dim=0)).nonzero(as_tuple=True)[0].tolist()
    g = torch.Generator().manual_seed(seed)
    rest = [i for i in torch.randperm(correct.shape[1], generator=g).tolist() if i not in set(disagree)]
    chosen = disagree[: max(n // 2, n - len(rest))]
    return chosen + rest[: n - len(chosen)], len(disagree)


def class_accuracy_report(runs, results, out_path=None, k=5):
    """Per-class accuracy of every run as text: top/bottom k per run, then all classes.

    birds525 has 5 test images per class, so accuracy takes only 6 values and the top
    and bottom are large tie groups; ties are broken alphabetically and the size of the
    tie group is printed so the k shown are not mistaken for a ranking.
    """
    names = list(runs)
    per_run = {n: {r["class"]: r for r in va.per_class_accuracy(runs[n], results[n])} for n in names}
    classes = sorted(set.intersection(*(set(v) for v in per_run.values())))
    width = max(len(c) for c in classes)

    def cell(row):
        return "{:d}/{:<d} {:5.1f}%".format(row["correct"], row["support"], row["accuracy"] * 100)

    lines = ["Per-class accuracy, {} split, {} classes".format(results[names[0]].split, len(classes))]
    for n in names:
        lines.append("  {:10s} {}  overall {:.2f}%".format(n, runs[n].name, results[n].accuracy * 100))

    for n in names:
        rows = [per_run[n][c] for c in classes]
        for title, ordered in (("top", sorted(rows, key=lambda r: (-r["accuracy"], r["class"]))),
                               ("bottom", sorted(rows, key=lambda r: (r["accuracy"], r["class"])))):
            edge = ordered[min(k, len(ordered)) - 1]["accuracy"]
            tied = sum(1 for r in rows if r["accuracy"] == edge)
            lines += ["", "{} -- {} {} (alphabetical within ties; {} classes at {:.0f}%)".format(
                n, title, k, tied, edge * 100)]
            lines += ["  {:{w}s}  {}".format(r["class"], cell(r), w=width) for r in ordered[:k]]

    if len(names) > 1:
        differ = [c for c in classes if len({per_run[n][c]["correct"] for n in names}) > 1]
        lines += ["", "classes where the runs differ: {} of {}".format(len(differ), len(classes))]
        lines.append("  {:{w}s}  {}".format("class", "  ".join("{:>12s}".format(n) for n in names), w=width))
        lines += ["  {:{w}s}  {}".format(c, "  ".join("{:>12s}".format(cell(per_run[n][c])) for n in names), w=width)
                  for c in differ]

    lines += ["", "all classes, alphabetical"]
    lines.append("  {:{w}s}  {}".format("class", "  ".join("{:>12s}".format(n) for n in names), w=width))
    lines += ["  {:{w}s}  {}".format(c, "  ".join("{:>12s}".format(cell(per_run[n][c])) for n in names), w=width)
              for c in classes]

    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


def stack(paths, out_path):
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.width for im in images)
    sheet = Image.new("RGB", (width, sum(im.height for im in images)), "white")
    y = 0
    for im in images:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR[:ANNOTATION_DIR]",
                   help="repeat per run; the annotation dir is the one that run was trained with")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=8, help="images in the comparison sheet")
    p.add_argument("--indices", type=int, nargs="*", default=None, help="explicit test-set indices instead")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--collage_n", type=int, default=10)
    p.add_argument("--tables_only", action="store_true", help="write class_accuracy.txt and stop")
    args = p.parse_args()

    runs, results, ann_dirs = {}, {}, {}
    for spec in args.run:
        name, _, rest = spec.partition("=")
        path, _, ann = rest.partition(":")
        runs[name] = va.load_run(resolve_run_dir(path))
        ann_dirs[name] = ann or None
        results[name] = va.evaluate(runs[name], split=args.split)
        print("{}: {}  accuracy {:.2f}%".format(name, runs[name], results[name].accuracy * 100))

    os.makedirs(args.out, exist_ok=True)
    table_path = os.path.join(args.out, "class_accuracy.txt")
    text = class_accuracy_report(runs, results, table_path)
    print(text.split("\nall classes")[0])
    print("wrote", table_path)
    if args.tables_only:
        return

    for name, run in runs.items():
        va.save_collages(run, results[name], out_dir=os.path.join(args.out, name),
                         n=args.collage_n, seed=args.seed)
        plt.close("all")

    if args.indices:
        indices = args.indices
    else:
        indices, n_disagree = pick_indices(results, args.n, args.seed)
        print("runs disagree on {} of {} images".format(n_disagree, len(next(iter(results.values())).labels)))
    print("comparison images:", indices)

    sheets = []
    for idx in indices:
        panels = []
        for name, run in runs.items():
            fig = va.explain_with_boxes(run, idx=idx, split=args.split, top_k=args.top_k,
                                        annotation_dir=ann_dirs[name])
            fig.suptitle("{}  --  image #{}".format(name, idx), fontsize=11, x=0.01, ha="left")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            path = os.path.join(args.out, name, "explain_{}.png".format(idx))
            fig.savefig(path, dpi=100)
            plt.close(fig)
            panels.append(path)
        sheet = os.path.join(args.out, "compare_{}.png".format(idx))
        stack(panels, sheet)
        sheets.append(sheet)

    stack(sheets, os.path.join(args.out, "comparison.png"))
    print("wrote {} and {} per-image sheets to {}".format("comparison.png", len(sheets), args.out))


if __name__ == "__main__":
    main()
