"""Carry Grounding DINO annotations over to the cleaned Birds-525 copy, without re-annotating.

Annotation files are named by image index in ImageFolder order (<split>/<idx>.json). The
cleaned copy (scripts/prepare_birds525.py) holds the same images, minus Looney Birds, in a
different order and with different file names, so each image is matched by content: MD5 of
the file bytes, falling back to a hash of the decoded, downsized pixels for any image whose
bytes differ. Every cleaned image must be matched; Looney Birds' annotations are dropped.

  python -m scripts.remap_annotations --old_root datasets/birds525 --old_ann annotations \
      --old birds525_train=train birds525_val=val birds525_valid=test \
      --new_root /shared/data/birds525 --new birds525c_train=train birds525c_valid=validation birds525c_val=test \
      --out annotations_birds525c
"""

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image


def image_files(root):
    """Files in torchvision ImageFolder order (sorted class folders, sorted files)."""
    out = []
    for c in sorted(e.name for e in os.scandir(root) if e.is_dir()):
        d = os.path.join(root, c)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    return out


def cls(path):
    """Class folder name, whitespace-normalised (the cleaned copy merges 'PARAKETT  AUKLET')."""
    return " ".join(os.path.basename(os.path.dirname(path)).split()).upper()


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def pixel_hash(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    return hashlib.md5((np.asarray(im) // 8).astype(np.uint8).tobytes()).hexdigest()


def hashes(files, fn, workers=16):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, files, chunksize=64))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old_root", required=True)
    p.add_argument("--old_ann", required=True)
    p.add_argument("--old", nargs="+", required=True, help="annotation_split=folder for the old copy")
    p.add_argument("--new_root", required=True)
    p.add_argument("--new", nargs="+", required=True, help="annotation_split=folder for the cleaned copy")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    old = {}                                                        # md5 -> annotation file
    old_px = {}
    for spec in args.old:
        ann, folder = spec.split("=")
        files = image_files(os.path.join(args.old_root, folder))
        n_ann = len([f for f in os.listdir(os.path.join(args.old_ann, ann)) if f.endswith(".json")])
        assert n_ann == len(files), f"{ann}: {n_ann} annotations vs {len(files)} images"
        for i, h in enumerate(hashes(files, md5)):                  # keyed by class too: a few images
            old[(cls(files[i]), h)] = os.path.join(args.old_ann, ann, f"{i}.json")   # repeat across classes
        print(f"old {ann} ({folder}): {len(files):,} images", flush=True)

    tmp = args.out + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    for spec in args.new:
        ann, folder = spec.split("=")
        files = image_files(os.path.join(args.new_root, folder))
        hs = hashes(files, md5)
        hs = [(cls(f), h) for f, h in zip(files, hs)]
        missing = [i for i, h in enumerate(hs) if h not in old]
        if missing and not old_px:                                  # bytes differ for some: match on pixels
            print(f"  {len(missing)} byte misses: building pixel hashes of the old copy", flush=True)
            for spec_o in args.old:
                a, f = spec_o.split("=")
                ofiles = image_files(os.path.join(args.old_root, f))
                for i, h in enumerate(hashes(ofiles, pixel_hash)):
                    old_px[(cls(ofiles[i]), h)] = os.path.join(args.old_ann, a, f"{i}.json")
        src = [old.get(h) for h in hs]
        for i in missing:
            src[i] = old_px.get((cls(files[i]), pixel_hash(files[i])))
        unmatched = [files[i] for i, s in enumerate(src) if s is None]
        assert not unmatched, f"{ann}: {len(unmatched)} images without annotations, e.g. {unmatched[:3]}"
        os.makedirs(os.path.join(tmp, ann))
        for i, s in enumerate(src):
            shutil.copyfile(s, os.path.join(tmp, ann, f"{i}.json"))
        print(f"new {ann} ({folder}): {len(files):,} images, all matched "
              f"({len(missing)} by pixels)", flush=True)
    shutil.rmtree(args.out, ignore_errors=True)
    os.rename(tmp, args.out)
    json.dump({"old": args.old, "new": args.new}, open(os.path.join(args.out, "remap.json"), "w"))
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
