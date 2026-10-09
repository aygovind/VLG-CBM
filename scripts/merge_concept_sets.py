"""Build the SAE and merged (pre-defined + SAE) concept files for a dataset.

Takes a flat SAE concept list plus its class -> concepts mapping (both produced
outside this repo, from SAE latent activations) and writes, next to the existing
pre-defined files:

    concept_files/<sae_name>_filtered.txt        deduplicated SAE concept set
    concept_files/<sae_name>_per_class.json      SAE concepts -> Grounding DINO prompt
    concept_files/<merged_name>_filtered.txt     pre-defined + non-overlapping SAE
    concept_files/<merged_name>_per_class.json   union of both per-class lists
    concept_files/<merged_name>_aliases.json     SAE concept -> pre-defined concept it
                                                 was folded into

Two concepts count as the same when the cosine similarity of their text embeddings
exceeds --threshold. Within the SAE set the earlier (higher-ranked) concept is kept;
between the two sets the pre-defined wording wins. The aliases file exists because
annotations made with the SAE prompts carry the SAE wording, so reusing them for the
merged model needs those labels renamed.

The class list is the base dataset's (<base>_classes.txt): the SAE and merged sets
are alternative concept sets for the same dataset, not new datasets.

Usage:
    python -m scripts.merge_concept_sets \
        --sae_filtered ~/Downloads/birds525sae_filtered.txt \
        --sae_per_class ~/Downloads/birds525sae_per_class.json
"""

import argparse
import json
import os

import torch

try:
    from data.utils import format_concept
except ImportError:  # data.utils pulls in the training stack; this script only needs the formatter

    def format_concept(s):
        s = s.lower()
        for ch in "-,.()":
            s = s.replace(ch, " ")
        if s[:2] == "a ":
            s = s[2:]
        elif s[:3] == "an ":
            s = s[3:]
        return " ".join(s.split())


def embed(texts, embedder, batch_size=256):
    """Unit-norm text embeddings. `mpnet` is what LF-CBM's concept filtering uses."""
    if embedder == "mpnet":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True).cpu()

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(embedder)
    tokenizer = open_clip.get_tokenizer(embedder)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            feats = model.encode_text(tokenizer(texts[i : i + batch_size]))
            out.append(torch.nn.functional.normalize(feats.float(), dim=-1))
    return torch.cat(out)


def unique(seq):
    return list(dict.fromkeys(seq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_filtered", type=str, required=True, help="flat SAE concept list, best first")
    parser.add_argument("--sae_per_class", type=str, required=True, help="json: class name -> SAE concepts")
    parser.add_argument("--base", type=str, default="birds525", help="dataset whose pre-defined files to merge with")
    parser.add_argument("--sae_name", type=str, default=None, help="defaults to <base>sae")
    parser.add_argument("--merged_name", type=str, default=None, help="defaults to <base>merged")
    parser.add_argument("--threshold", type=float, default=0.85, help=">= 1 disables similarity matching")
    parser.add_argument("--sae_only", action="store_true", help="write the SAE files only, skip the merged set")
    parser.add_argument(
        "--embedder", type=str, default="mpnet",
        help="'mpnet' (sentence-transformers all-mpnet-base-v2) or an open_clip model id, e.g. hf-hub:imageomics/bioclip")
    parser.add_argument("--concept_dir", type=str, default="concept_files")
    parser.add_argument("--output_dir", type=str, default=None, help="defaults to --concept_dir")
    args = parser.parse_args()
    sae_name = args.sae_name or f"{args.base}sae"
    merged_name = args.merged_name or f"{args.base}merged"
    output_dir = args.output_dir or args.concept_dir

    with open(os.path.join(args.concept_dir, f"{args.base}_classes.txt")) as f:
        classes = [c for c in f.read().split("\n") if c.strip()]
    with open(os.path.join(args.concept_dir, f"{args.base}_filtered.txt")) as f:
        base = unique(format_concept(c) for c in f.read().split("\n") if c.strip())
    with open(os.path.join(args.concept_dir, f"{args.base}_per_class.json")) as f:
        base_per_class = {c: [format_concept(x) for x in v] for c, v in json.load(f).items()}

    with open(os.path.expanduser(args.sae_filtered), encoding="utf-8") as f:
        sae = unique(format_concept(c) for c in f.read().split("\n") if c.strip())
    with open(os.path.expanduser(args.sae_per_class), encoding="utf-8") as f:
        raw = json.load(f)
    # the SAE files key classes by the dataset's folder names; ours are lower-cased and
    # keep the folders' irregular spacing, so match on a normalized form
    def norm(c):
        return " ".join(c.lower().replace("_", " ").split())

    by_key = {norm(k): v for k, v in raw.items()}
    unknown = sorted(set(by_key) - {norm(c) for c in classes})
    if unknown:
        raise SystemExit(f"{len(unknown)} classes in {args.sae_per_class} are not in {args.base}_classes.txt: {unknown[:5]}")
    missing = [c for c in classes if norm(c) not in by_key]
    sae_per_class = {c: [format_concept(x) for x in by_key.get(norm(c), [])] for c in classes}
    stray = sorted({x for v in sae_per_class.values() for x in v} - set(sae))
    if stray:
        raise SystemExit(f"{len(stray)} per-class SAE concepts are not in {args.sae_filtered}: {stray[:5]}")

    match = args.threshold < 1
    if match:
        sae_emb = embed(sae, args.embedder)

    # 1. near-duplicates inside the SAE set: greedy, so the earlier concept survives
    kept, folded = [], {}
    for i in range(len(sae)):
        if match and kept:
            sims = sae_emb[kept] @ sae_emb[i]
            best = int(sims.argmax())
            if sims[best] > args.threshold:
                folded[sae[i]] = sae[kept[best]]
                print(f"dup     {sims[best]:.3f}  {sae[i]!r} -> {sae[kept[best]]!r}")
                continue
        kept.append(i)

    sae_per_class = {c: unique(folded.get(x, x) for x in v) for c, v in sae_per_class.items()}
    # a concept no class prompt mentions can never be annotated (see build_concept_files)
    reachable = {x for v in sae_per_class.values() for x in v}
    unreachable = [sae[i] for i in kept if sae[i] not in reachable]
    kept = [i for i in kept if sae[i] in reachable]
    sae_final = [sae[i] for i in kept]

    os.makedirs(output_dir, exist_ok=True)

    def write(name, concepts, per_class):
        with open(os.path.join(output_dir, f"{name}_filtered.txt"), "w") as f:
            f.write("\n".join(concepts))
        with open(os.path.join(output_dir, f"{name}_per_class.json"), "w") as f:
            json.dump(per_class, f, indent=1)

    def sizes(per_class):
        s = [len(v) for v in per_class.values()]
        return f"min {min(s)}  mean {sum(s) / len(s):.1f}  max {max(s)}"

    write(sae_name, sae_final, sae_per_class)
    print(f"embedder: {args.embedder if match else 'none'}  threshold: {args.threshold}")
    print(f"SAE: {len(sae)} in, {len(folded)} folded as near-duplicates, "
          f"{len(unreachable)} dropped as unreachable -> {len(sae_final)}")
    print(f"concepts per class  {sae_name}: {sizes(sae_per_class)}")

    if not args.sae_only:
        # 2. overlap with the pre-defined set: the pre-defined wording wins
        aliases = {}
        if match:
            sims = sae_emb[kept] @ embed(base, args.embedder).T
            best_sim, best_idx = sims.max(dim=1)
            for concept, s, j in zip(sae_final, best_sim.tolist(), best_idx.tolist()):
                if s > args.threshold:
                    aliases[concept] = base[j]
                    print(f"overlap {s:.3f}  {concept!r} -> {base[j]!r}")
        else:
            aliases = {c: c for c in sae_final if c in set(base)}
        merged = base + [c for c in sae_final if c not in aliases]
        merged_per_class = {
            c: unique(base_per_class.get(c, []) + [aliases.get(x, x) for x in sae_per_class[c]]) for c in classes
        }
        write(merged_name, merged, merged_per_class)
        with open(os.path.join(output_dir, f"{merged_name}_aliases.json"), "w") as f:
            json.dump(aliases, f, indent=1)
        print(f"merged: {len(base)} pre-defined + {len(sae_final) - len(aliases)} SAE "
              f"({len(aliases)} overlapping) = {len(merged)}")
        print(f"concepts per class  {merged_name}: {sizes(merged_per_class)}")
    if missing:
        print(f"WARNING: {len(missing)} classes have no SAE concepts: {missing[:5]}")


if __name__ == "__main__":
    main()
