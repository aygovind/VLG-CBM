"""Build annotations for a merged concept set from two existing annotation passes.

A merged concept set (scripts/merge_concept_sets.py) is the union of two sets that
have each already been through Grounding DINO, so its annotations are the union of
the two passes -- nothing has to be detected again. For every image this writes

    <out>/<split>/<idx>.json = <base>/<split>/<idx>.json
                             + the detections of <extra>/<split>/<idx>.json,
                               relabelled through the aliases file

The relabelling is what makes the union correct: detections made under a concept that
the merge folded into another must count for the concept that survived, or they are
silently lost when ConceptDataset matches labels against the merged concept list.

Note this is the union of two prompts, not one pass with both concept lists in the
prompt; Grounding DINO scores each phrase in the context of its prompt, so the two
are close but not identical. The upside is that the merged model is supervised by
exactly the detections the two single-set models were.

Existing outputs are skipped, so an interrupted run resumes.

Usage:
    python -m scripts.merge_annotations --base annotations --extra annotations_sae \
        --aliases concept_files/birds525merged_aliases.json --out annotations_merged \
        --splits birds525_train birds525_val birds525_valid
"""

import argparse
import json
import os

from tqdm import tqdm

from data.utils import format_concept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="annotation dir whose labels are kept as they are")
    parser.add_argument("--extra", type=str, required=True, help="annotation dir whose labels go through --aliases")
    parser.add_argument("--aliases", type=str, required=True, help="json: concept in --extra -> merged concept")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", required=True)
    args = parser.parse_args()

    with open(args.aliases) as f:
        aliases = json.load(f)

    for split in args.splits:
        base_dir, extra_dir, out_dir = (os.path.join(d, split) for d in (args.base, args.extra, args.out))
        names = sorted(n for n in os.listdir(base_dir) if n.endswith(".json"))
        extra_names = {n for n in os.listdir(extra_dir) if n.endswith(".json")}
        if set(names) != extra_names:
            raise SystemExit(f"{split}: {base_dir} and {extra_dir} do not cover the same images "
                             f"({len(names)} vs {len(extra_names)})")
        os.makedirs(out_dir, exist_ok=True)
        done = set(os.listdir(out_dir))

        renamed = written = 0
        for name in tqdm(names, desc=split):
            if name in done:
                continue
            with open(os.path.join(base_dir, name)) as f:
                merged = json.load(f)
            with open(os.path.join(extra_dir, name)) as f:
                extra = json.load(f)
            # entry 0 is the image path, the same in both
            for box in extra[1:]:
                target = aliases.get(format_concept(box["label"]))
                if target is not None:
                    box["label"] = target
                    renamed += 1
                merged.append(box)
            # a json that exists must be complete, since that is what resuming relies on
            tmp = os.path.join(out_dir, name + ".part")
            with open(tmp, "w") as f:
                json.dump(merged, f)
            os.replace(tmp, os.path.join(out_dir, name))
            written += 1
        print(f"{split}: {len(names)} images, {written} written, {len(names) - written} already there, "
              f"{renamed} detections relabelled")


if __name__ == "__main__":
    main()
