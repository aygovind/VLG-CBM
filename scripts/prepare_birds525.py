"""Write the canonical, cleaned copy of Birds-525 that every experiment should use.

Source: the Hugging Face mirror of the Kaggle dataset (yashikota/birds-525-species-
image-classification) at a pinned revision. Two fixes are applied:

  1. "LOONEY BIRDS" is dropped from every split: its images are cartoons/people, not birds.
  2. Class names are whitespace-normalised, which merges "PARAKETT  AUKLET" (double
     space, holds the images) with "PARAKETT AUKLET" -- the same species split in two.

Output is a plain ImageFolder tree (torchvision.datasets.ImageFolder reads it directly;
label = position in sorted classes.txt) with the original JPEG bytes, unmodified:

  <out>/{train,validation,test}/<CLASS NAME>/<hf row index>.jpg
  <out>/classes.txt   one name per line, in label order
  <out>/index.csv     split, hf_index, hf_label, hf_class, label, class, path
  <out>/manifest.json source, revision, counts, what was removed or merged
  <out>/README.md

Written to <out>.part and renamed at the end, so a half-written copy is never mistaken
for a finished one; an existing <out>/manifest.json means done, and the script exits.

Usage:
  python scripts/prepare_birds525.py --out /shared/data/birds525
  python scripts/prepare_birds525.py --out /tmp/b --parquet_dir local_parquets  # offline test
"""

import argparse
import csv
import glob
import io
import json
import os
import shutil
import time
from collections import Counter

import pyarrow.parquet as pq

REPO = "yashikota/birds-525-species-image-classification"
REVISION = "6064cca293f31ad3ed9b5dc1267b39e0a0764afd"
SPLITS = ("train", "validation", "test")
DROP = "LOONEY BIRDS"


def download(revision, dest):
    """The dataset's parquet files at a pinned revision."""
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", revision=revision, local_dir=dest,
                      allow_patterns=["data/*.parquet", "README.md"])
    return os.path.join(dest, "data")


def class_names(parquet_file):
    """ClassLabel names stored in the parquet's Hugging Face metadata."""
    meta = json.loads(pq.read_schema(parquet_file).metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def readme(manifest):
    m = manifest
    counts = "\n".join(f"| {s} | {m['images'][s]:,} | {m['removed_images'][s]} |" for s in SPLITS)
    merged = "\n".join(f"  - {' + '.join(repr(n) for n in v)} -> `{k}`" for k, v in m["merged_classes"].items())
    return f"""# Birds-525 (cleaned) -- use this copy

Canonical copy of Birds-525 for all experiments. Do **not** use the raw Kaggle or Hugging
Face download, or older copies on other volumes (e.g. `vlgcbm-workspace:VLG-CBM/datasets`):
they still contain the classes removed here.

## Changes from the source
1. **"LOONEY BIRDS" removed** from all splits. Its images are cartoons and people, not
   birds: noise for any bird classifier or concept model.
2. **Class names whitespace-normalised**, which merges duplicate classes:
{merged}
   Other names lose double spaces only (e.g. "COCK OF THE  ROCK" -> "COCK OF THE ROCK").

Result: **{m['num_classes']} classes** (source: {m['source_num_classes']} names).

| split | images | removed (Looney Birds) |
|---|---|---|
{counts}

## Layout
ImageFolder: `<split>/<CLASS NAME>/<hf row index>.jpg`, original JPEG bytes unmodified.
Labels = position in `classes.txt` (sorted names), which is also what
`torchvision.datasets.ImageFolder` assigns. `index.csv` maps every image back to its
source row and source label.

```python
from torchvision.datasets import ImageFolder
train = ImageFolder("{m['path']}/train", transform=...)
```

## Source
{m['source']} @ `{m['revision']}` (mirror of https://www.kaggle.com/datasets/gpiosenka/100-bird-species).
Written {m['created']} by `VLG-CBM/scripts/prepare_birds525.py`.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--parquet_dir", default=None, help="use these parquet files instead of downloading")
    args = p.parse_args()
    out = args.out.rstrip("/")
    if os.path.exists(os.path.join(out, "manifest.json")):
        print(f"{out} already prepared, skipping")
        return

    tmp = out + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    started = time.time()
    if args.parquet_dir:
        src = args.parquet_dir
    else:
        print(f"downloading {REPO} @ {args.revision[:10]}")
        src = download(args.revision, os.path.join(tmp, "_download"))

    files = {s: sorted(glob.glob(os.path.join(src, f"{s}-*.parquet"))) for s in SPLITS}
    files = {s: f for s, f in files.items() if f}
    names = class_names(next(iter(files.values()))[0])
    clean = [" ".join(n.split()) for n in names]
    classes = sorted(set(c for c in clean if c != DROP))
    label_of = {c: i for i, c in enumerate(classes)}
    dups = Counter(c for c in clean if c != DROP)
    merged = {c: [n for n in names if " ".join(n.split()) == c] for c, k in dups.items() if k > 1}
    print(f"{len(names)} source names -> {len(classes)} classes; dropped {DROP!r}; merged {merged}")

    rows, images, removed, reencoded = [], Counter(), Counter(), 0
    for split, paths in files.items():
        for c in classes:
            os.makedirs(os.path.join(tmp, split, c), exist_ok=True)
        hf_index = 0
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["image", "label"]):
                for image, hf_label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                    name = clean[hf_label]
                    if name == DROP:
                        removed[split] += 1
                    else:
                        data = image["bytes"]
                        if data[:2] != b"\xff\xd8":  # not a JPEG: re-encode so every file is one
                            from PIL import Image
                            buf = io.BytesIO()
                            Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=95)
                            data, reencoded = buf.getvalue(), reencoded + 1
                        rel = os.path.join(split, name, f"{hf_index:05d}.jpg")
                        with open(os.path.join(tmp, rel), "wb") as f:
                            f.write(data)
                        rows.append([split, hf_index, hf_label, names[hf_label], label_of[name], name, rel])
                        images[split] += 1
                    hf_index += 1
        print(f"  {split}: {images[split]:,} images written, {removed[split]} Looney Birds removed "
              f"({time.time() - started:.0f}s)")

    for split in files:
        empty = [c for c in classes if not os.listdir(os.path.join(tmp, split, c))]
        assert not empty, f"{split}: classes with no images: {empty[:5]}"
    with open(os.path.join(tmp, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(tmp, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "hf_index", "hf_label", "hf_class", "label", "class", "path"])
        w.writerows(rows)
    manifest = {"path": out, "source": f"huggingface.co/datasets/{REPO}", "revision": args.revision,
                "source_num_classes": len(names), "num_classes": len(classes),
                "removed_class": DROP, "removed_images": {s: removed[s] for s in files},
                "merged_classes": merged, "images": {s: images[s] for s in files},
                "reencoded_non_jpeg": reencoded, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write(readme(manifest))
    shutil.rmtree(os.path.join(tmp, "_download"), ignore_errors=True)
    with open(os.path.join(tmp, "manifest.json"), "w") as f:  # last: marks the copy complete
        json.dump(manifest, f, indent=2)
    shutil.rmtree(out, ignore_errors=True)  # an unfinished copy (no manifest) from an older run
    os.rename(tmp, out)
    print(f"done in {time.time() - started:.0f}s: {out}  {json.dumps(manifest['images'])}")


if __name__ == "__main__":
    main()
