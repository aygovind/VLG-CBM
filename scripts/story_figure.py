"""The qualitative comparison grid from compare_vlgcbm_models_colab.ipynb, as a script.

One image per failure mode -- models disagree most, strong model right, domain model
right, every model wrong -- with one column per run showing its prediction, confidence
and top concepts. The figure is drawn by vlgcbm_analysis.story_figure, the function the
notebook calls, so it is the same image.

--flag chooses what the cell colours mean:
  none         green = correct, red = wrong. Nothing else is judged.
  vlm          also amber ("OK*") for a correct prediction where a concept the model
               leveraged is not visible in the image, as judged by a small VLM
               (vlgcbm_analysis.VLMJudge; ~4.4GB download on first use; a GPU by
               default, or --vlm_device cpu at ~20-40s per judge call).
               The judge is sanity-checked first and refused if it cannot tell a
               detected concept from one taken from another image.
  sufficiency  amber when the displayed top concepts alone do not reproduce the
               prediction -- the notebook's default proxy, no VLM involved.

Runs are named on the command line; the names become the column headers.

Usage:
  python scripts/story_figure.py --out figures/story.png --flag none \
      --run sae=saved_models/birds525sae_v2_vit_in21k \
      --run control=saved_models/birds525_v2_vit_in21k \
      --run merged=saved_models/birds525merged_v2_vit_in21k
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display on the pod; must precede any pyplot import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlgcbm_analysis as va  # noqa: E402
from scripts.qualitative_report import resolve_run_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR[:ANNOTATION_DIR]",
                   help="repeat per run, in column order; the annotation dir is only read "
                        "for the first run, by the VLM sanity check")
    p.add_argument("--out", required=True, help="path of the png to write")
    p.add_argument("--flag", choices=["none", "vlm", "sufficiency"], default="none")
    p.add_argument("--split", default="val")
    p.add_argument("--top_concepts", type=int, default=2, help="concepts listed per cell, and judged")
    p.add_argument("--strong", default=None, help="run name for the 'strong' rows; default: most accurate")
    p.add_argument("--domain", default=None, help="run name for the 'domain' rows; default: least accurate")
    p.add_argument("--vlm_model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--vlm_threshold", type=float, default=0.5)
    p.add_argument("--vlm_device", default=None,
                   help="default: cuda when available; cpu works for one figure (~20-40s per call)")
    args = p.parse_args()

    names, dirs, ann_dirs = [], [], []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        path, _, ann = rest.partition(":")
        names.append(name)
        dirs.append(resolve_run_dir(path))
        ann_dirs.append(ann or va.DEFAULT_ANNOTATION_DIR)

    outs = va.evaluate_many(dirs, split=args.split, keep_concept_acts=True)
    for o, name in zip(outs, names):
        o.name = name  # column headers, and what --strong / --domain refer to

    flag_fn, flag_mode = None, None
    if args.flag == "sufficiency":
        flag_mode = "sufficiency"
    elif args.flag == "vlm":
        # the judge needs raw images and one run's detections for its sanity check
        run = va.load_run(dirs[0])
        res = va.evaluate(run, split=args.split)
        os.environ["VLGCBM_ANNOTATION_DIR"] = ann_dirs[0]
        import torch
        device = args.vlm_device or ("cuda" if torch.cuda.is_available() else "cpu")
        judge = va.VLMJudge(args.vlm_model, device=device)
        stats = judge.sanity_check(run, res, split=args.split, n=6)
        if stats["separation"] <= 0.05:
            sys.exit("judge does not discriminate (separation {:+.3f}); its flags would be noise. "
                     "Try another --vlm_model or use --flag none.".format(stats["separation"]))
        cache = {}
        flag_fn = va.vlm_flag_fn(judge, run, split=args.split, k=args.top_concepts,
                                 threshold=args.vlm_threshold, cache=cache, verbose=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig = va.story_figure(outs, split=args.split, strong=args.strong, domain=args.domain,
                          top_concepts=args.top_concepts, flag_mode=flag_mode, flag_fn=flag_fn,
                          save_path=args.out)
    if fig is None:
        sys.exit("no figure: the runs never disagree and never all fail on this split")
    if args.flag == "vlm":
        print("{} judge calls".format(len(cache)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
