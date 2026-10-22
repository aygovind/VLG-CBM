"""Name patch-SAE latents with a VLM (GPT-6 Luna), from vit-sae-latent-naming.ipynb.

For each latent the VLM sees 10 positive panels (original | activation overlay | the
3x3-patch region around the peak, enlarged) and 10 negatives (5 visually similar images
where the latent is off, 5 random), and returns a structured hypothesis: status, concept,
region, evidence, alternative, verification test. Works from a results file of
train_patch_sae.py / sweep_patch_sae.py (results/test_results.pt) and its patch tokens,
on CPU.

Every response is cached under out/responses/latent_<id>_<signature>.json (signature =
model + prompt + schema + rendered images), so reruns cost nothing and an interrupted
run resumes. Summary: out/names.csv (one row per latent, with the class make-up of the
positives, never shown to the VLM, for a later class-detector filter).

  python -m scripts.name_latents --results R.pt --patches P.npy --out naming --random 20 --dry_run
  OPENAI_API_KEY=... python -m scripts.name_latents --results R.pt --patches P.npy --out naming --random 20
"""

import argparse
import base64
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from scripts.latent_collage import latent_maps, load_results

MODEL = "gpt-6-luna"
PRICE = {"input": 0.10, "cached": 0.01, "output": 0.50}       # USD per million tokens (check current rates)
N_POS, N_HARD, N_RANDOM = 10, 5, 5
OFF = 0.05          # a latent is "off" in an image if its max activation < OFF x its max over the test set

PROMPT = """
You are interpreting a sparse autoencoder latent from a vision model trained on bird photos.

You receive 10 positive examples P1-P10 and 10 negative examples N1-N10.
Each positive panel contains:
  left: original image;
  middle: activation overlay, with brighter regions indicating stronger activation;
  right: the region around the strongest patch, enlarged.
Negative panels are original images where the latent is essentially inactive.
N1-N5 were chosen to look similar to the positives; N6-N10 are random.

Infer the visible concept that best distinguishes positives from negatives.
Use both the original images and the highlighted regions. The overlay is a clue:
late-layer tokens can encode information from elsewhere in the image.
Do not interpret the overlay's artificial colors as properties of the bird.
The enlarged region adds no new visual information.

Do not use species names, dataset class names, or latent IDs as concept names.
Keep the concept label to at most 5 words.
Prefer a short, concrete visible trait. Consider background explanations too.
Do not invent a shared feature if examples are inconsistent.
Cite example IDs in your evidence. Include a competing interpretation.
Return status "coherent", "mixed", or "unclear".
Return region: where the concept lies — "bird part" (a specific part or marking),
"whole bird" (overall shape, colour or posture), "background" (scenery, perch, water,
foliage), "image frame" (borders, corners, position in the photo) or "mixed".
Your proposed name is a hypothesis, not a verified label.
""".strip()

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["coherent", "mixed", "unclear"]},
        "concept": {"type": "string"},
        "region": {"type": "string", "enum": ["bird part", "whole bird", "background", "image frame", "mixed"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "alternative": {"type": "string"},
        "verification_test": {"type": "string"},
    },
    "required": ["status", "concept", "region", "evidence", "alternative", "verification_test"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- examples

def select_examples(r, latent, n_pos=N_POS, n_hard=N_HARD, n_random=N_RANDOM, off=OFF, seed=42):
    """Top-n_pos images by max activation; negatives where the latent is off (below
    off x its max): n_hard most CLS-similar to the positives, n_random random. Exact
    zeros are rare for an L1 SAE, so "off" uses a small relative threshold."""
    scores = r["scores"][:, latent]
    pos = scores.topk(n_pos).indices
    assert (scores[pos] > 0).all(), f"latent {latent} has fewer than {n_pos} positive images"
    neg = torch.where(scores < off * scores.max())[0]
    assert len(neg) >= n_hard + n_random, f"latent {latent} is on in almost every image"
    cls = F.normalize(r["cls"].float(), dim=-1)
    sim = (cls[neg] @ cls[pos].T).max(1).values
    hard = neg[sim.topk(n_hard).indices]
    rest = neg[~torch.isin(neg, hard)]
    rand = rest[torch.randperm(len(rest), generator=torch.Generator().manual_seed(seed + latent))[:n_random]]
    return pos.tolist(), hard.tolist() + rand.tolist()


def image(r, i):
    return Image.open(io.BytesIO(r["images_jpeg_224"][i])).convert("RGB")


def labeled(images, labels, label_h=24):
    """Images side by side with a text label above each."""
    w, h = images[0].size
    panel = Image.new("RGB", (w * len(images), h + label_h), "white")
    draw = ImageDraw.Draw(panel)
    for j, (im, text) in enumerate(zip(images, labels)):
        panel.paste(im, (j * w, label_h))
        draw.text((j * w + 4, 5), text, fill="black")
    return panel


def positive_panel(r, i, amap, vmax, number):
    """original | overlay (transparent where the latent is off) | enlarged 3x3 region at the peak."""
    orig = image(r, i)
    w, h = orig.size
    g = amap.shape[0]
    heat = np.asarray(Image.fromarray((amap / vmax).astype("float32")).resize((w, h), Image.Resampling.NEAREST)).clip(0, 1)
    rgb = np.asarray(orig).astype("float32")
    color = matplotlib.colormaps["magma"](heat)[..., :3] * 255
    alpha = 0.65 * heat[..., None]
    over = Image.fromarray(((1 - alpha) * rgb + alpha * color).astype("uint8"))
    r0, c0 = divmod(int(amap.argmax()), g)
    box = (int(max(c0 - 1, 0) * w / g), int(max(r0 - 1, 0) * h / g),
           int(min(c0 + 2, g) * w / g), int(min(r0 + 2, g) * h / g))
    crop = orig.crop(box).resize((w, h), Image.Resampling.LANCZOS)
    return labeled([orig, over, crop], [f"P{number}: original", "activation overlay", "region at peak"])


def render(r, latent, pos, neg):
    """Panels exactly as sent to the VLM: 10 positive, 10 negative."""
    maps = latent_maps(r, pos, latent)
    vmax = max(float(maps.max()), 1e-8)
    positives = [positive_panel(r, i, m, vmax, n) for n, (i, m) in enumerate(zip(pos, maps), start=1)]
    negatives = [labeled([image(r, i)], [f"N{n}: {'similar' if n <= N_HARD else 'random'} negative"])
                 for n, i in enumerate(neg, start=1)]
    return positives, negatives


def contact_sheet(panels, columns):
    w, h = panels[0].size
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w, rows * h), "white")
    for k, p in enumerate(panels):
        sheet.paste(p, ((k % columns) * w, (k // columns) * h))
    return sheet


# ---------------------------------------------------------------- request

def data_url(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_request(panels):
    """Responses-API content list plus a cache signature of everything that affects the answer."""
    urls = [data_url(p) for p in panels]
    content = [{"type": "input_text", "text": "Interpret the following examples."}]
    content += [{"type": "input_image", "image_url": u, "detail": "high"} for u in urls]
    sig = hashlib.sha256(json.dumps({"model": MODEL, "prompt": PROMPT, "schema": SCHEMA, "images": urls,
                                     "reasoning": "none", "max_output_tokens": 700},
                                    sort_keys=True).encode()).hexdigest()[:16]
    return content, sig


def image_tokens(panels, multiplier=1.2, patch=32):
    """Estimated image input tokens for patch-based models: ceil(w/32) x ceil(h/32) x 1.2."""
    return sum(int(np.ceil(np.ceil(p.width / patch) * np.ceil(p.height / patch) * multiplier)) for p in panels)


def cost(usage):
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return ((usage["input_tokens"] - cached) * PRICE["input"] + cached * PRICE["cached"]
            + usage["output_tokens"] * PRICE["output"]) / 1e6


def call_vlm(client, content):
    """One structured request; the raw response is returned for saving before parsing."""
    t0 = time.perf_counter()
    resp = client.responses.create(
        model=MODEL, instructions=PROMPT, input=[{"role": "user", "content": content}],
        reasoning={"effort": "none"},
        text={"format": {"type": "json_schema", "name": "latent_interpretation", "strict": True, "schema": SCHEMA}},
        max_output_tokens=700, service_tier="default", store=False)
    return {"response_id": resp.id, "model": resp.model, "status": resp.status, "output_text": resp.output_text,
            "usage": resp.usage.model_dump(), "api_seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- one latent / many latents

def name_latent(r, latent, out, client=None, dry_run=False):
    """Select examples, render panels, call the VLM (or load the cached answer). Returns a
    summary row. dry_run: render and save the panels and estimate tokens, without an API call."""
    pos, neg = select_examples(r, latent)
    positives, negatives = render(r, latent, pos, neg)
    content, sig = build_request(positives + negatives)
    os.makedirs(os.path.join(out, "panels"), exist_ok=True)
    os.makedirs(os.path.join(out, "responses"), exist_ok=True)
    contact_sheet(positives, 2).save(os.path.join(out, "panels", f"latent_{latent}_positives.png"))
    contact_sheet(negatives, 5).save(os.path.join(out, "panels", f"latent_{latent}_negatives.png"))
    labels = r["labels"][pos].tolist()
    row = {"latent": latent, "signature": sig, "est_image_tokens": image_tokens(positives + negatives),
           "pos_classes": len(set(labels)), "pos_max_same_class": max(labels.count(c) for c in set(labels)),
           "positive_images": pos, "negative_images": neg}
    if dry_run:
        return row
    path = os.path.join(out, "responses", f"latent_{latent}_{sig}.json")
    cached = os.path.exists(path)
    saved = json.load(open(path)) if cached else call_vlm(client, content)
    if not cached:
        json.dump(saved, open(path, "w"), indent=2)                    # saved before parsing: no silent retries
    row.update(cached=cached, api_status=saved["status"], input_tokens=saved["usage"]["input_tokens"],
               output_tokens=saved["usage"]["output_tokens"], cost_usd=0.0 if cached else cost(saved["usage"]),
               api_seconds=0.0 if cached else saved["api_seconds"])
    if saved["status"] == "completed":
        ans = json.loads(saved["output_text"])
        row.update(status=ans["status"], concept=ans["concept"], region=ans["region"],
                   evidence=" | ".join(ans["evidence"]), alternative=ans["alternative"],
                   verification_test=ans["verification_test"])
    return row


def name_latents(r, latents, out, client=None, workers=4, dry_run=False, log=print):
    """Name many latents in parallel; writes out/names.csv after every finished latent."""
    rows, failures, t0 = [], [], time.time()
    csv = os.path.join(out, "names.csv" if not dry_run else "dry_run.csv")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(name_latent, r, lat, out, client, dry_run): lat for lat in latents}
        for k, fut in enumerate(as_completed(futures), start=1):
            lat = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:                                      # one failure does not stop the run
                failures.append({"latent": lat, "error": repr(e)})
            pd.DataFrame(rows).sort_values("latent").to_csv(csv, index=False)
            if k % max(1, len(latents) // 10) == 0 or k == len(latents):
                spent = sum(x.get("cost_usd", 0) for x in rows)
                log(f"  {k}/{len(latents)} latents, {len(failures)} failed, ${spent:.4f}, {time.time() - t0:.0f}s")
    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(out, "failures.csv"), index=False)
    return pd.DataFrame(rows).sort_values("latent").reset_index(drop=True), failures


def eligible_latents(r, n_pos=N_POS, n_neg=N_HARD + N_RANDOM, off=OFF):
    """Latents with n_pos positive images and n_neg images where they are off."""
    s = r["scores"]
    return torch.where(((s > 0).sum(0) >= n_pos) & ((s < off * s.amax(0)).sum(0) >= n_neg))[0].tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="results/test_results.pt")
    p.add_argument("--patches", default=None, help="local test_patches.npy (default: the path stored in results)")
    p.add_argument("--out", required=True)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--latents", type=int, nargs="+")
    who.add_argument("--random", type=int, help="this many random eligible latents")
    who.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry_run", action="store_true", help="render panels and estimate tokens; no API calls")
    args = p.parse_args()

    r = load_results(args.results, args.patches)
    ok = eligible_latents(r)
    if args.latents:
        latents = args.latents
    elif args.random:
        latents = sorted(np.random.default_rng(args.seed).choice(ok, size=args.random, replace=False).tolist())
    else:
        latents = ok
    print(f"{len(ok):,}/{r['scores'].shape[1]:,} latents eligible; naming {len(latents)}: "
          f"{latents if len(latents) <= 30 else str(latents[:30]) + ' ...'}")
    client = None
    if not args.dry_run:
        from openai import OpenAI
        assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY first"
        client = OpenAI(timeout=180, max_retries=2)
    df, fails = name_latents(r, latents, args.out, client, args.workers, args.dry_run)
    if args.dry_run:
        tok = df["est_image_tokens"].mean() + 350
        print(f"estimated input tokens per latent ~{tok:,.0f} -> ~${tok * PRICE['input'] / 1e6 * len(ok):.2f} "
              f"input cost for all {len(ok):,} eligible latents (+ output)")
    else:
        done = df.dropna(subset=["concept"]) if "concept" in df else df.iloc[:0]
        print(f"named {len(done)}/{len(latents)}; cost ${df.get('cost_usd', pd.Series([0])).sum():.4f}; "
              f"mean API time {df.get('api_seconds', pd.Series([0])).mean():.1f}s")
        if len(done):
            print(done[["latent", "status", "region", "concept", "pos_classes"]].to_string(index=False))
            print("\nstatus:", done["status"].value_counts().to_dict(), "| region:", done["region"].value_counts().to_dict())


if __name__ == "__main__":
    main()
