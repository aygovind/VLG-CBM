"""Grounding DINO concept annotations with an arbitrary concept list per image.

scripts/generate_annotations.py prompts every image of a class with that class's
concept list, which is how VLG-CBM's training labels are made. This generalises it to
any concept list per image, so the same detector can be pointed at a test set for
evaluation:

  --per_class_file   json {class name: [concepts]}  -- training-style annotations
  --per_image_file   json {image index: [concepts]}
  --bundle           an eval bundle (scripts/generate_eval.py) -> each image gets the
                     concepts its true class and its predicted class use (non-zero
                     final-layer weight): "is what the decision rests on visible?"

Long lists are split into several prompts that each fit Grounding DINO's 256-token
window (one prompt silently truncates), and the detections are merged. Output is the
format ConceptDataset and vlgcbm_analysis.annotations_for read:

  <output_dir>/<dataset>/<idx>.json = [{"img_path": ...}, {"label", "logit", "box", "logits"}, ...]

Settings default to the training pass: the class name leads each prompt (it tightens
the boxes; drop it with --no_class_name) and text_threshold is 0.15. A written .json
is a finished image, so a rerun resumes.

Usage:
  python -m scripts.annotate_concepts --dataset cub_val \
      --bundle /shared/vlgcbm_eval_paper/cub.pt --output_dir annotations_cub_decision
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

import data.utils as utils
from scripts.generate_annotations import (Resize, T, get_predictions, load_annotation_model,
                                          process_annotations, save_annotations)


def lists_from_bundle(path):
    """Per image: concepts with a non-zero weight for its true or its predicted class."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    used = b["W_g"] != 0
    out = {}
    for i, (t, p) in enumerate(zip(b["labels"].tolist(), b["preds"].tolist())):
        idx = (used[t] | used[p]).nonzero(as_tuple=True)[0].tolist()
        out[i] = [b["concepts"][c] for c in idx]
    return out


def chunk(concepts, prefix, tokenizer, max_tokens):
    """Split concepts into prompts "prefix . c1 . c2 ." that each fit max_tokens."""
    prompts, cur = [], []
    for c in concepts:
        trial = " . ".join(prefix + cur + [c]) + " ."
        if cur and len(tokenizer(trial)["input_ids"]) > max_tokens:
            prompts.append(" . ".join(prefix + cur) + " .")
            cur = []
        cur.append(c)
    if cur:
        prompts.append(" . ".join(prefix + cur) + " .")
    return prompts


class Jobs(torch.utils.data.Dataset):
    """One item per (image, prompt) pair; images are loaded on demand."""

    def __init__(self, dataset, jobs):
        self.dataset, self.jobs = dataset, jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, j):
        return self.dataset[self.jobs[j][0]][0], j


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="e.g. cub_val (the test set), cub_train")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--per_class_file")
    src.add_argument("--per_image_file")
    src.add_argument("--bundle")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_class_name", action="store_true", help="do not lead each prompt with the class name")
    p.add_argument("--text_threshold", type=float, default=0.15)
    p.add_argument("--max_tokens", type=int, default=250, help="per prompt; Grounding DINO's window is 256")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="first N images only, for a quick test")
    p.add_argument("--config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    p.add_argument("--checkpoint", default="GroundingDINO/groundingdino_swinb_cogcoor.pth")
    args = p.parse_args()

    classes = utils.get_classes(args.dataset.split("_")[0])
    pil_data = utils.get_data(args.dataset, preprocess=None)
    transform = T.Compose([Resize(800), T.RandomResize([800]), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = utils.get_data(args.dataset, preprocess=lambda x: transform(x, None)[0])
    targets = list(dataset.targets)
    n = len(targets) if args.limit is None else min(args.limit, len(targets))

    if args.bundle:
        lists = lists_from_bundle(args.bundle)
        if len(lists) != len(targets):
            raise SystemExit(f"bundle has {len(lists)} images, {args.dataset} has {len(targets)}")
    elif args.per_image_file:
        lists = {int(k): v for k, v in json.load(open(args.per_image_file)).items()}
    else:
        per_class = json.load(open(args.per_class_file))
        lists = {i: per_class[classes[t]] for i, t in enumerate(targets)}

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    todo = [i for i in range(n) if i in lists and not os.path.exists(os.path.join(out_dir, f"{i}.json"))]
    print(f"{args.dataset}: {n} images, {n - len(todo)} already annotated, {len(todo)} to do")
    if not todo:
        return

    model, tokenizer = load_annotation_model(args.config, args.checkpoint, device=args.device)

    # one job per (image, prompt); a concept that survives formatting differently from
    # how the tokenizer decodes it could never be matched later, so report those up front
    jobs, mismatched = [], set()
    for i in todo:
        concepts = list(dict.fromkeys(utils.format_concept(c) for c in lists[i]))
        for c in concepts:
            if utils.format_concept(tokenizer.decode(tokenizer(c)["input_ids"][1:-1])) != c:
                mismatched.add(c)
        prefix = [] if args.no_class_name else [utils.format_concept(classes[targets[i]])]
        for prompt in chunk(concepts, prefix, tokenizer, args.max_tokens):
            jobs.append((i, prompt))
    per_image = {}
    for i, _ in jobs:
        per_image[i] = per_image.get(i, 0) + 1
    print(f"{len(jobs)} prompts, {len(jobs) / len(todo):.2f} per image; "
          f"{len(mismatched)} concepts will never match their decoded label: {sorted(mismatched)[:10]}")

    loader = torch.utils.data.DataLoader(Jobs(dataset, jobs), batch_size=args.batch_size,
                                         num_workers=args.num_workers, shuffle=False, pin_memory=True)
    found = {}
    for images, js in tqdm(loader):
        prompts = [jobs[j][1] for j in js.tolist()]
        logits, boxes = get_predictions(model, images.to(args.device), prompts)
        for k, j in enumerate(js.tolist()):
            i, prompt = jobs[j]
            anns = process_annotations(pil_data[i][0], prompt, logits[k].cpu().numpy(),
                                       boxes[k].cpu().numpy(), tokenizer, args.text_threshold)
            found.setdefault(i, []).extend(anns)
            per_image[i] -= 1
            if per_image[i] == 0:  # every prompt of this image is done
                img_path = pil_data.samples[i][0] if hasattr(pil_data, "samples") else None
                save_annotations(out_dir, f"{i}.json", found.pop(i), img_path)

    print(f"wrote {len(todo)} annotation files to {out_dir}")


if __name__ == "__main__":
    main()
