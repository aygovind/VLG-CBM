"""Build VLG-CBM concept files for a dataset from the LF-CBM concept-set artifacts.

VLG-CBM needs three files per dataset:

    concept_files/<dataset>_classes.txt     class names, in ImageFolder (sorted) order
    concept_files/<dataset>_filtered.txt    the flat concept set the CBL predicts
    concept_files/<dataset>_per_class.json  concepts -> Grounding DINO prompt, per class

LF-CBM only produces the first two. The per-class file is what
scripts.generate_annotations turns into a detection prompt, so it is rebuilt here by
intersecting the filtered concept set with the raw GPT-3 proposals for each class --
the same mapping model.utils.get_per_class_filtered_concepts does for the datasets
that ship with VLG-CBM.

A concept in the filtered set that reaches no class prompt can never be annotated, so
it would be silently dropped at train time by get_filtered_concepts_and_counts. The
coverage number printed at the end is the check for that.

Usage:
    python -m scripts.build_concept_files --dataset birds525 --lfcbm_dir ../Label-free-CBM
"""

import argparse
import json
import os

from data.utils import format_concept

# LF-CBM's GPT-3 concept proposals come in three flavours; a class's raw concept pool
# is their union, matching how the filtered set was originally produced.
GPT3_KINDS = ("important", "around", "superclass")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lfcbm_dir", type=str, required=True, help="path to the Label-free-CBM checkout")
    parser.add_argument("--classes_file", type=str, default=None, help="defaults to <lfcbm_dir>/data/<dataset>.txt")
    parser.add_argument(
        "--filtered_file", type=str, default=None,
        help="defaults to <lfcbm_dir>/data/concept_sets/<dataset>_filtered.txt")
    parser.add_argument("--output_dir", type=str, default="concept_files")
    parser.add_argument("--name", type=str, default=None, help="output dataset name; defaults to --dataset")
    parser.add_argument(
        "--max_classes", type=int, default=None,
        help="keep only the first N classes -- used to build the smoke dataset")
    args = parser.parse_args()
    name = args.name or args.dataset

    classes_file = args.classes_file or os.path.join(args.lfcbm_dir, "data", f"{args.dataset}.txt")
    filtered_file = args.filtered_file or os.path.join(
        args.lfcbm_dir, "data", "concept_sets", f"{args.dataset}_filtered.txt")

    with open(classes_file) as f:
        all_classes = [c for c in f.read().split("\n") if c.strip()]
    classes = all_classes[: args.max_classes] if args.max_classes is not None else all_classes
    with open(filtered_file) as f:
        filtered = [format_concept(c) for c in f.read().split("\n") if c.strip()]
    filtered = list(dict.fromkeys(filtered))

    # union of the GPT-3 proposals, keyed by class
    raw_per_class = {c: [] for c in classes}
    for kind in GPT3_KINDS:
        path = os.path.join(args.lfcbm_dir, "data", "concept_sets", "gpt3_init", f"gpt3_{args.dataset}_{kind}.json")
        if not os.path.exists(path):
            print(f"skipping missing {path}")
            continue
        with open(path) as f:
            proposals = json.load(f)
        for cls, concepts in proposals.items():
            if cls not in raw_per_class:
                # classes dropped by --max_classes are expected to be missing here;
                # a class the classes file has never heard of is not
                if cls not in all_classes:
                    print(f"warning: class {cls!r} in {kind} proposals is not in {classes_file}")
                continue
            raw_per_class[cls].extend(format_concept(c) for c in concepts)

    per_class = {}
    for cls in classes:
        pool = set(raw_per_class[cls])
        per_class[cls] = [c for c in filtered if c in pool]

    # A concept no class prompt mentions can never be annotated, so writing it into the
    # concept set would only get it dropped again by get_filtered_concepts_and_counts.
    # For the full dataset every concept is reachable and this is a no-op; it matters
    # when --max_classes carves out a subset.
    covered = {c for concepts in per_class.values() for c in concepts}
    unreachable = [c for c in filtered if c not in covered]
    filtered = [c for c in filtered if c in covered]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{name}_classes.txt"), "w") as f:
        f.write("\n".join(classes))
    with open(os.path.join(args.output_dir, f"{name}_filtered.txt"), "w") as f:
        f.write("\n".join(filtered))
    with open(os.path.join(args.output_dir, f"{name}_per_class.json"), "w") as f:
        json.dump(per_class, f, indent=1)

    sizes = [len(v) for v in per_class.values()]
    print(f"wrote concept_files/{name}_{{classes.txt,filtered.txt,per_class.json}}")
    print(f"classes: {len(classes)}  concepts: {len(filtered)}  dropped as unreachable: {len(unreachable)}")
    print(f"concepts per class: min {min(sizes)}  mean {sum(sizes) / len(sizes):.1f}  max {max(sizes)}")
    empty = [c for c, v in per_class.items() if not v]
    if empty:
        print(f"WARNING: {len(empty)} classes have no concepts, e.g. {empty[:5]}")


if __name__ == "__main__":
    main()
