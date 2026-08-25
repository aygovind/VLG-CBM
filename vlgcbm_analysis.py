"""Reusable analysis helpers for trained VLG-CBM run directories.

The counterpart to cbm_analysis.py in the Label-free-CBM repo, and deliberately the
same API: load_run / evaluate / compare / plot_wrong_predictions / explain_example /
sankey_static / concept_heatmap / confused_pairs / result_collage / plot_class_accuracy
/ plot_confusion / story_figure all behave identically, so the two sets of results can
be read side by side.

Only the loading layer differs. LF-CBM saves one fused model (W_c, W_g, b_g, proj_mean,
proj_std); VLG-CBM saves four pieces that have to be reassembled -- backbone, cbl.pt,
the normalisation statistics, and final.pt. VLGCBM below stitches them back together
behind the same `model(x) -> (logits, concept_activations)` interface the analysis
functions expect, which is why everything downstream of it is unchanged.

Three additions have no LF-CBM equivalent, because they need the Grounding DINO
annotations: annotations_for, explain_with_boxes and concept_agreement. Those are the
ones worth reaching for when the question is whether a concept is actually grounded in
the image rather than merely correlated with the class.

Run from the repo root with DATASET_FOLDER set:

    export DATASET_FOLDER=/workspace/VLG-CBM/datasets
    export VLGCBM_BIOCLIP_CKPT=/workspace/models/bioclip/open_clip_pytorch_model.bin
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import utils as data_utils
from model.cbm import Backbone, BackboneCLIP, ConceptLayer, FinalLayer, NormalizationLayer

# train_cbm.py writes all of these; a directory missing any of them is a run that died
# before finishing rather than one worth loading.
ARTIFACTS = ("cbl.pt", "final.pt", "train_concept_features_mean.pt",
             "train_concept_features_std.pt", "concepts.txt", "args.txt")

DEFAULT_ANNOTATION_DIR = "annotations"


class VLGCBM(nn.Module):
    """The four saved pieces of a VLG-CBM run, reassembled into one model.

    Returns (logits, concept_activations) to match the LF-CBM CBM_model interface the
    analysis functions were written against. The activations are post-normalisation,
    which is the representation the final layer actually consumes -- so
    `concept_act * final.weight[c]` really is the contribution to class c's logit,
    and every contribution plot below is reading the model rather than approximating it.
    """

    def __init__(self, backbone, cbl, normalization, final):
        super().__init__()
        self.backbone = backbone
        self.cbl = cbl
        self.normalization = normalization
        self.final = final

    def forward(self, x):
        concept_act = self.normalization(self.cbl(self.backbone(x)))
        return self.final(concept_act), concept_act


class Run:
    """A loaded CBM run: model plus the labels needed to interpret its outputs."""

    def __init__(self, load_dir, model, concepts, classes, train_args, preprocess, device):
        self.load_dir = load_dir
        self.model = model
        self.concepts = concepts
        self.classes = classes
        self.train_args = train_args
        self.preprocess = preprocess
        self.device = device

    @property
    def name(self):
        """Something readable for plot titles and comparison rows.

        train_cbm.py names the run directory <dataset>_cbm_<timestamp>, which says
        nothing about which model it is; the parent -- saved_models/birds525_bioclip --
        is the part worth showing. Falls back to the basename for a flat layout.
        """
        base = os.path.basename(self.load_dir.rstrip("/"))
        if base.startswith("{}_cbm_".format(self.train_args.get("dataset", ""))):
            parent = os.path.basename(os.path.dirname(self.load_dir.rstrip("/")))
            if parent:
                return parent
        return base

    @property
    def dataset(self):
        return self.train_args["dataset"]

    @property
    def backbone(self):
        return self.train_args["backbone"]

    def __repr__(self):
        return "<Run {} backbone={} concepts={} classes={}>".format(
            self.name, self.backbone, len(self.concepts), len(self.classes))


def _read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.read().split("\n") if line.strip()]


def _load_classes(load_dir, dataset, n_expected):
    """VLG-CBM does not copy the class list into the run dir, so read the label file.

    Warn on a length mismatch rather than silently mislabelling every plot -- a trailing
    newline in a label file inflates the class count by one and shifts every name.
    """
    classes = _read_lines(data_utils.LABEL_FILES[dataset])
    if len(classes) != n_expected:
        print("WARNING: {} class names but the final layer has {} outputs -- "
              "labels may be misaligned".format(len(classes), n_expected))
    return classes


def _load_backbone(train_args, device):
    if train_args["backbone"].startswith("clip_"):
        return BackboneCLIP(train_args["backbone"],
                            use_penultimate=train_args.get("use_clip_penultimate", False),
                            device=device)
    return Backbone(train_args["backbone"], train_args["feature_layer"], device)


def load_run(load_dir, device=None):
    """Load a run directory produced by train_cbm.py."""
    missing = [f for f in ARTIFACTS if not os.path.exists(os.path.join(load_dir, f))]
    if missing:
        raise FileNotFoundError("{} is not a complete run dir; missing {}".format(
            load_dir, missing))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        train_args = json.load(f)

    concepts = _read_lines(os.path.join(load_dir, "concepts.txt"))
    backbone = _load_backbone(train_args, device)

    # ConceptLayer.from_pretrained re-derives its width from args.txt, which does not
    # record how many concepts survived filtering; build it from concepts.txt instead.
    cbl = ConceptLayer(backbone.output_dim, len(concepts),
                       num_hidden=train_args.get("cbl_hidden_layers", 0), device=device)
    cbl.load_state_dict(torch.load(os.path.join(load_dir, "cbl.pt"), map_location=device))

    normalization = NormalizationLayer.from_pretrained(load_dir, device=device)

    # Size the final layer from the checkpoint rather than from the label file, so the
    # class names are checked against what the model actually outputs instead of
    # deciding it.
    final_state = torch.load(os.path.join(load_dir, "final.pt"), map_location=device)
    classes = _load_classes(load_dir, train_args["dataset"], final_state["weight"].shape[0])
    final = FinalLayer(len(concepts), final_state["weight"].shape[0], device=device)
    final.load_state_dict(final_state)

    model = VLGCBM(backbone, cbl, normalization, final).to(device).eval()
    return Run(load_dir, model, concepts, classes, train_args, backbone.preprocess, device)


def run_dir(model, dataset="birds525", save_dir="saved_models"):
    """Resolve a model name to its newest complete run directory.

        run = va.load_run(va.run_dir("bioclip"))

    `model` is the suffix used by the configs -- bioclip, vit_in21k, dino, clip_vitb16,
    rn50, bioclip2 -- so saved_models/<dataset>_<model>/<newest timestamped run>.
    Raises with the available names rather than an opaque path error.
    """
    parent = os.path.join(save_dir, "{}_{}".format(dataset, model))
    if os.path.isdir(parent):
        if _is_run_dir(parent):
            return parent
        nested = sorted(os.path.join(parent, s) for s in os.listdir(parent)
                        if os.path.isdir(os.path.join(parent, s))
                        and _is_run_dir(os.path.join(parent, s)))
        if nested:
            return nested[-1]
    raise FileNotFoundError("no complete run for model={!r} dataset={!r} under {}; "
                            "available: {}".format(model, dataset, save_dir,
                                                   ", ".join(list_models(dataset, save_dir)) or "none"))


def list_models(dataset="birds525", save_dir="saved_models"):
    """Model names that have a complete run, for the config cell to print."""
    prefix = "{}_".format(dataset)
    out = []
    for d in sorted(os.listdir(save_dir)) if os.path.isdir(save_dir) else []:
        if not d.startswith(prefix) or d.endswith("_standard"):
            continue
        try:
            run_dir(d[len(prefix):], dataset, save_dir)
        except FileNotFoundError:
            continue
        out.append(d[len(prefix):])
    return out


# ImageFolder rescans the directory tree on construction, which on the PVC is slow
# enough to notice -- and the plotting helpers each ask for the split again. Keyed by
# the run so two backbones with different preprocessing do not share an entry.
_DATA_CACHE = {}


def get_data(run, split="val", raw=False):
    """Eval split for this run. raw=True returns undecoded PIL images for display.

    Note VLG-CBM's naming: `<dataset>_val` is the held-out test set. The validation
    split used during training is carved out of `<dataset>_train` and has no name here,
    so split="val" is the set train_cbm.py reports test accuracy on.
    """
    # raw images are backbone-independent, so every run can share one entry
    key = (run.dataset, split, raw, None if raw else run.load_dir)
    if key not in _DATA_CACHE:
        name = "{}_{}".format(run.dataset, split)
        if raw:
            import torchvision.transforms as T
            _DATA_CACHE[key] = data_utils.get_data(name, preprocess=T.Lambda(lambda x: x))
        else:
            _DATA_CACHE[key] = data_utils.get_data(name, run.preprocess)
    return _DATA_CACHE[key]


def annotation_dir_for(run, split="val", annotation_dir=None):
    root = annotation_dir or os.environ.get("VLGCBM_ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR)
    return os.path.join(root, "{}_{}".format(run.dataset, split))


def annotations_for(run, idx, split="val", annotation_dir=None, threshold=None):
    """Grounding DINO boxes for one image, as (label, logit, box) dicts.

    Defaults to the confidence threshold the run was trained with, so what you see is
    what the CBL was actually supervised on rather than the raw detector output.
    """
    if threshold is None:
        threshold = run.train_args.get("cbl_confidence_threshold", 0.15)
    path = os.path.join(annotation_dir_for(run, split, annotation_dir), "{}.json".format(int(idx)))
    if not os.path.exists(path):
        raise FileNotFoundError("no annotation at {}".format(path))
    with open(path) as f:
        data = json.load(f)
    return [a for a in data[1:] if a["logit"] > threshold]


class Results:
    """Per-example predictions and concept activations, in dataset order."""

    def __init__(self, run, split, preds, labels, concept_acts, logits=None):
        self.run = run
        self.split = split
        self.preds = preds
        self.labels = labels
        self.concept_acts = concept_acts
        self.logits = logits

    @property
    def accuracy(self):
        return (self.preds == self.labels).float().mean().item()

    @property
    def correct(self):
        return self.preds == self.labels

    @property
    def wrong_indices(self):
        return (self.preds != self.labels).nonzero(as_tuple=True)[0]

    @property
    def confidence(self):
        """Softmax probability of the predicted class, per example.

        Ranking by this is what separates "the model was sure and right" from
        "sure and wrong" -- the second group is where the interesting failures are.
        """
        if self.logits is None:
            raise RuntimeError("no logits stored; re-run evaluate() to populate them")
        return torch.softmax(self.logits, dim=1).max(dim=1).values

    def __repr__(self):
        return "<Results {} {} acc={:.4f} n={}>".format(
            self.run.name, self.split, self.accuracy, len(self.labels))


def evaluate(run, split="val", batch_size=256, num_workers=4, cache=True):
    """Run the model over a split, keeping predictions and concept activations.

    The result is cached next to the run as eval_<split>.pt, because nothing here
    depends on anything but the weights and the split -- so re-running the cell, or
    reopening the notebook tomorrow, should not re-run the backbone over the whole set.
    Pass cache=False to force recomputation.
    """
    cache_path = os.path.join(run.load_dir, "eval_{}.pt".format(split))
    if cache and os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("n_concepts") == len(run.concepts):
            return Results(run, split, blob["preds"], blob["labels"],
                           blob["concept_acts"], blob["logits"])
        print("note: cached {} predates the current model; recomputing".format(cache_path))

    data = get_data(run, split)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    preds, labels, acts, all_logits = [], [], [], []
    with torch.no_grad():
        for images, y in loader:
            logits, concept_act = run.model(images.to(run.device))
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(y)
            acts.append(concept_act.cpu())
            all_logits.append(logits.cpu())

    results = Results(run, split, torch.cat(preds), torch.cat(labels),
                      torch.cat(acts), torch.cat(all_logits))
    if cache:
        torch.save({"preds": results.preds, "labels": results.labels,
                    "concept_acts": results.concept_acts, "logits": results.logits,
                    "n_concepts": len(run.concepts)}, cache_path)
    return results


def class_indices(results, class_name, only=None, n=None, predicted=False):
    """Dataset indices for one class, ready to hand to explain_example().

        idx = ca.class_indices(res, "abbotts babbler", only="wrong")[0]
        ca.explain_example(run, idx)

    Args:
        class_name: exact string from run.classes. Matching is exact rather than
            fuzzy -- birds525 has near-duplicate names, and a substring match would
            silently pick the wrong species.
        only: None for every example, "correct" or "wrong" to keep just those.
        n: keep at most this many (they come back in dataset order, not shuffled).
        predicted: match on what the model *predicted* instead of ground truth.
            "wrong" + predicted=True is what you want to see the images being
            mistaken for this class, rather than the ones being missed.
    """
    run = results.run
    if class_name not in run.classes:
        near = [c for c in run.classes if class_name.lower() in c.lower()]
        raise ValueError("{!r} not in classes{}".format(
            class_name, "; did you mean {}?".format(near[:5]) if near else ""))

    target = run.classes.index(class_name)
    match = (results.preds if predicted else results.labels) == target
    if only == "correct":
        match = match & results.correct
    elif only == "wrong":
        match = match & ~results.correct
    elif only is not None:
        raise ValueError("only must be None, 'correct' or 'wrong'")

    idx = match.nonzero(as_tuple=True)[0].tolist()
    return idx[:n] if n else idx


def summarize(run, results=None, split="val"):
    """One-row summary dict; use with compare() across runs."""
    results = results or evaluate(run, split)
    W_g = run.model.final.weight.detach().cpu()
    nnz = (W_g.abs() > 1e-5).sum().item()
    return {
        "run": run.name,
        "backbone": run.backbone,
        "clip_name": run.train_args["clip_name"],
        "lam": run.train_args["lam"],
        "accuracy": round(results.accuracy, 4),
        "n_concepts": len(run.concepts),
        "frac_non_zero": round(nnz / W_g.numel(), 4),
        "concepts_per_class": round(nnz / W_g.shape[0], 1),
    }


def compare(load_dirs, split="val", device=None):
    """Summary table across several runs. Returns a DataFrame if pandas is available."""
    rows = []
    for d in load_dirs:
        try:
            rows.append(summarize(load_run(d, device), split=split))
        except Exception as exc:
            print("skipping {}: {}".format(d, exc))
    try:
        import pandas as pd
        return pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    except ImportError:
        return rows


def _is_run_dir(path):
    return all(os.path.exists(os.path.join(path, f)) for f in ARTIFACTS)


def find_runs(save_dir="saved_models", newest_only=True):
    """Every complete run directory under save_dir.

    VLG-CBM nests one level deeper than LF-CBM: train_cbm.py writes
    saved_models/<config save_dir>/<dataset>_cbm_<timestamp>/, so the run itself is a
    grandchild of save_dir. Both layouts are accepted, and by default only the newest
    run under each parent is returned -- reruns of the same config otherwise stack up
    and quietly turn a five-backbone comparison into a nine-row table.

    The *_standard directories are skipped for free: black-box runs have no cbl.pt.
    """
    if not os.path.isdir(save_dir):
        return []

    found = []
    for d in sorted(os.listdir(save_dir)):
        parent = os.path.join(save_dir, d)
        if not os.path.isdir(parent):
            continue
        if _is_run_dir(parent):                      # LF-CBM-style flat layout
            found.append(parent)
            continue
        nested = sorted(os.path.join(parent, s) for s in os.listdir(parent)
                        if os.path.isdir(os.path.join(parent, s))
                        and _is_run_dir(os.path.join(parent, s)))
        if nested:
            found.extend(nested[-1:] if newest_only else nested)
    return found


# --------------------------------------------------------------------- plotting

def _normalization(preprocess):
    """Recover (mean, std) from a preprocessing pipeline so images can be un-normalised.

    Each backbone family normalises differently -- CLIP, augreg_in21k and DINO all
    use different constants -- so hardcoding ImageNet values would tint the displayed
    images for most models.
    """
    from torchvision import transforms as T
    stack, seen = [preprocess], 0
    while stack and seen < 100:
        seen += 1
        t = stack.pop()
        if isinstance(t, T.Normalize):
            return np.array(t.mean), np.array(t.std)
        for attr in ("transforms", "transform"):
            sub = getattr(t, attr, None)
            if isinstance(sub, (list, tuple)):
                stack.extend(sub)
            elif sub is not None:
                stack.append(sub)
    print("note: no Normalize found in preprocess; showing images un-scaled")
    return np.zeros(3), np.ones(3)


def _to_displayable(img, preprocess):
    mean, std = _normalization(preprocess)
    arr = img.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(arr * std + mean, 0, 1)


def _top_contributions(run, concept_act, class_idx, top_k=8):
    """Concept contributions to one class logit, largest magnitude first."""
    weights = run.model.final.weight[class_idx].detach().cpu()
    contrib = (concept_act.detach().cpu() * weights).numpy()
    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    labels = [("NOT " if concept_act[i] < 0 else "") + run.concepts[i] for i in order]
    return labels, contrib[order]


def plot_wrong_predictions(run, results, n=5, top_k=8, seed=None):
    """Misclassified examples: image alongside the concepts that drove the wrong call."""
    import matplotlib.pyplot as plt

    wrong = results.wrong_indices
    if len(wrong) == 0:
        print("no wrong predictions")
        return
    if seed is not None:
        torch.manual_seed(seed)
    chosen = wrong[torch.randperm(len(wrong))[:n]].tolist()

    data = get_data(run, results.split)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 3.2 * len(chosen)))
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(chosen):
        img, _ = data[idx]
        pred_i, true_i = int(results.preds[idx]), int(results.labels[idx])

        axes[row, 0].imshow(_to_displayable(img, run.preprocess))
        axes[row, 0].axis("off")
        axes[row, 0].set_title("#{}  true: {}\npredicted: {}".format(
            idx, run.classes[true_i], run.classes[pred_i]), fontsize=9)

        labels, values = _top_contributions(run, results.concept_acts[idx], pred_i, top_k)
        colors = ["tab:red" if v > 0 else "tab:blue" for v in values]
        axes[row, 1].barh(range(len(values))[::-1], values, color=colors)
        axes[row, 1].set_yticks(range(len(values))[::-1])
        axes[row, 1].set_yticklabels(labels, fontsize=8)
        axes[row, 1].axvline(0, color="k", lw=0.8)
        axes[row, 1].set_xlabel("contribution to predicted class", fontsize=8)

    plt.tight_layout()
    plt.show()
    print("{} wrong of {} ({:.2f}% accuracy)".format(
        len(wrong), len(results.labels), results.accuracy * 100))


def explain_example(run, idx, split="val", top_k=8):
    """Single-example view: image, top-2 predictions, concept contribution bars."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    data = get_data(run, split)
    raw = get_data(run, split, raw=True)

    x, true_i = data[idx]
    with torch.no_grad():
        logits, concept_act = run.model(x.unsqueeze(0).to(run.device))

    probs = torch.nn.functional.softmax(logits[0], dim=0)
    top_vals, top_classes = torch.topk(logits[0], k=min(2, len(run.classes)))

    try:
        display(raw[idx][0].resize([320, 320]))
    except Exception:
        plt.imshow(_to_displayable(x, run.preprocess)); plt.axis("off"); plt.show()

    print("#{}  true: {}".format(idx, run.classes[int(true_i)]))
    for rank, c in enumerate(top_classes.tolist()):
        print("  {}. {}  logit {:.3f}  p={:.3f}".format(
            rank + 1, run.classes[c], top_vals[rank], probs[c]))

    labels, values = _top_contributions(run, concept_act[0], int(top_classes[0]), top_k)
    colors = ["tab:red" if v > 0 else "tab:blue" for v in values]
    plt.figure(figsize=(7, 0.4 * len(values) + 1))
    plt.barh(range(len(values))[::-1], values, color=colors)
    plt.yticks(range(len(values))[::-1], labels, fontsize=9)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("contribution to {}".format(run.classes[int(top_classes[0])]))
    plt.tight_layout()
    plt.show()


def sankey(run, class_a, class_b=None, weight_cutoff=0.05, max_per_class=12,
           class_colors=None, concept_colors=None, dominance=0.85):
    """Interactive version of sankey_static(), same content and colouring.

    Kept for hovering over ribbons to read exact weights; sankey_static() is the one
    that produces the paper-style figure to drop into a write-up.
    """
    import plotly.graph_objects as go

    classes = _as_class_list(run, class_a, class_b)
    edges = _sankey_edges(run, classes, weight_cutoff, max_per_class)
    if not edges:
        print("no concepts above |weight| > {}; lower weight_cutoff".format(weight_cutoff))
        return None

    concept_labels, flows, colors = _sankey_order_and_colors(
        edges, classes, class_colors, concept_colors, dominance)
    nodes = concept_labels + list(classes)
    index = {label: i for i, label in enumerate(nodes)}
    class_color = _class_colors(classes, class_colors)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=12, thickness=14,
                  color=[colors[l] for l in concept_labels] + class_color,
                  line=dict(width=0)),
        link=dict(
            source=[index[e[0]] for e in edges],
            target=[len(concept_labels) + e[1] for e in edges],
            value=[e[2] for e in edges],
            color=[_rgba(colors[e[0]], 0.55) for e in edges],
        ),
    ))
    fig.update_layout(
        title=dict(text=_sankey_title(run, classes).replace("\n", "<br>"), x=0.5,
                   xanchor="center"),
        font=dict(family="DejaVu Sans, Arial", size=12),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(400, 22 * len(concept_labels)))
    fig.show()
    return fig


def concept_heatmap(run, results, n_examples=30, n_concepts=20):
    """Which concepts fire across many examples -- dataset-level companion to the bars."""
    import matplotlib.pyplot as plt

    acts = results.concept_acts
    top = torch.argsort(acts.abs().mean(dim=0), descending=True)[:n_concepts]
    subset = acts[:n_examples][:, top].numpy()
    scale = np.abs(subset).max() or 1.0

    plt.figure(figsize=(10, 8))
    plt.imshow(subset, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    plt.colorbar(label="concept activation")
    plt.yticks(range(min(n_examples, len(subset))),
               ["#{} ({})".format(i, run.classes[int(results.labels[i])])
                for i in range(min(n_examples, len(subset)))], fontsize=7)
    plt.xticks(range(len(top)), [run.concepts[i] for i in top], rotation=90, fontsize=7)
    plt.title("{} -- most active concepts".format(run.name))
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------- static sankey
#
# Styled after Figure 3 of the Label-free CBM paper (Oikarinen et al., ICLR 2023),
# which was drawn by hand in sankeymatic.com: concepts on the left under a "Concept"
# header, the two classes on the right under "Prediction", labels sitting inside the
# plot next to their node, ribbon width = |w|, and ribbons taking the colour of their
# source concept. The paper colours concepts by semantic group manually; the default
# here colours each concept by the class it feeds, which needs no hand-editing -- pass
# `concept_colors` to reproduce the paper's semantic grouping exactly.

CLASS_PALETTE = ["#e0872f", "#2f9e8f", "#8b6bbf", "#c0587a", "#5b8fd4", "#b5a642"]
NEUTRAL_CONCEPT = "#9a9a9a"          # concepts that feed both classes about equally
LABEL_CHIP = "#f2f2f2"               # chip behind labels that sit on a thick ribbon


def _as_class_list(run, classes, class_b=None):
    """Accept ("A", "B") or ["A", "B"], and fail loudly on a name that is not a class."""
    if isinstance(classes, str):
        classes = [classes]
    classes = list(classes)
    if class_b is not None:
        classes.append(class_b)
    for name in classes:
        if name not in run.classes:
            raise ValueError("{!r} not in classes; e.g. {}".format(name, run.classes[:5]))
    return classes


def _sankey_edges(run, classes, weight_cutoff, max_per_class):
    """(concept_label, class_position, |w|) for each surviving final-layer weight.

    Negative weights become "NOT <concept>" and flow at |w|, which is the paper's
    convention: the diagram shows how strongly a concept decides the class, not the
    sign of the decision.
    """
    final_weight = run.model.final.weight.detach().cpu()
    edges = []
    for pos, class_name in enumerate(classes):
        weights = final_weight[run.classes.index(class_name)]
        keep = (weights.abs() > weight_cutoff).nonzero(as_tuple=True)[0]
        keep = keep[torch.argsort(weights[keep].abs(), descending=True)]
        if max_per_class:
            keep = keep[:max_per_class]
        for ci in keep.tolist():
            w = weights[ci].item()
            label = run.concepts[ci] if w > 0 else "NOT {}".format(run.concepts[ci])
            edges.append((label, pos, abs(w)))
    return edges


def _class_colors(classes, class_colors=None):
    if isinstance(class_colors, dict):
        return [class_colors.get(c, CLASS_PALETTE[i % len(CLASS_PALETTE)])
                for i, c in enumerate(classes)]
    if class_colors:
        return [class_colors[i % len(class_colors)] for i in range(len(classes))]
    return [CLASS_PALETTE[i % len(CLASS_PALETTE)] for i in range(len(classes))]


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "rgba({},{},{},{})".format(r, g, b, alpha)


def _sankey_order_and_colors(edges, classes, class_colors, concept_colors, dominance):
    """Stack order and node colour for every concept.

    Concepts are grouped by the class they feed most so each class's band stays
    contiguous -- the paper's layout, and the only thing that keeps the ribbons from
    crossing into a tangle. A concept that splits its flow between both classes goes
    grey and sorts after the class it leans toward, which is what makes the shared
    concepts (the reason two classes are confusable) visible at a glance.
    """
    palette = _class_colors(classes, class_colors)
    flow, per_class = {}, {}
    for label, pos, w in edges:
        flow[label] = flow.get(label, 0.0) + w
        by_class = per_class.setdefault(label, {})
        by_class[pos] = by_class.get(pos, 0.0) + w

    home, colors = {}, {}
    for label, by_class in per_class.items():
        pos, share = max(by_class.items(), key=lambda kv: kv[1])
        home[label] = pos
        colors[label] = palette[pos] if share / flow[label] >= dominance else NEUTRAL_CONCEPT
    if concept_colors:
        for label in colors:
            override = concept_colors.get(label, concept_colors.get(
                label[4:] if label.startswith("NOT ") else label))
            if override:
                colors[label] = override

    order = sorted(flow, key=lambda l: (home[l],
                                        colors[l] == NEUTRAL_CONCEPT,
                                        -flow[l]))
    return order, flow, colors


def _sankey_title(run, classes):
    return "{} CBM\n{}".format(run.dataset, " vs ".join(classes))


def _save_fig(fig, save_path, dpi=150):
    """Write a figure, creating the parent directory so notebook paths just work."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("saved", save_path)


def _flow_path(x0, x1, top0, bot0, top1, bot1):
    """Cubic-Bezier ribbon joining a slice of a left node to a slice of a right one."""
    from matplotlib.path import Path
    mid = (x0 + x1) / 2.0
    verts = [(x0, top0),
             (mid, top0), (mid, top1), (x1, top1),
             (x1, bot1),
             (mid, bot1), (mid, bot0), (x0, bot0),
             (x0, top0)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    return Path(verts, codes)


def _draw_sankey(ax, run, classes, weight_cutoff=0.05, max_per_class=12, gap=0.012,
                 class_colors=None, concept_colors=None, dominance=0.85,
                 fontsize=11, title=None, headers=True, chip_min_height=None):
    """Draw one paper-style panel into `ax`. Returns the concept labels drawn."""
    import matplotlib.patches as patches

    classes = _as_class_list(run, classes)
    edges = _sankey_edges(run, classes, weight_cutoff, max_per_class)
    if not edges:
        print("no concepts above |weight| > {}; lower weight_cutoff".format(weight_cutoff))
        return []

    concept_labels, concept_flow, colors = _sankey_order_and_colors(
        edges, classes, class_colors, concept_colors, dominance)
    class_color = _class_colors(classes, class_colors)

    class_flow = {}
    for _, pos, w in edges:
        class_flow[pos] = class_flow.get(pos, 0.0) + w

    def _stack(keys, flows, node_gap):
        """Assign each node a [top, bottom] band, normalised to fill height 1."""
        total = sum(flows[k] for k in keys)
        span = 1.0 - node_gap * max(len(keys) - 1, 0)
        spans, y = {}, 1.0
        for k in keys:
            h = span * flows[k] / total if total else 0.0
            spans[k] = [y, y - h]
            y -= h + node_gap
        return spans

    left = _stack(concept_labels, concept_flow, gap)
    right = _stack(list(range(len(classes))), class_flow, gap * 2)

    x_left, x_right, node_w = 0.015, 0.985, 0.011
    pad = 0.012

    # ribbons first, so nodes and labels draw over their ends
    left_cursor = {k: left[k][0] for k in left}
    right_cursor = {k: right[k][0] for k in right}
    for label, pos, w in sorted(edges, key=lambda e: concept_labels.index(e[0])):
        scale_l = (left[label][0] - left[label][1]) / concept_flow[label]
        scale_r = (right[pos][0] - right[pos][1]) / class_flow[pos]
        t0, b0 = left_cursor[label], left_cursor[label] - w * scale_l
        t1, b1 = right_cursor[pos], right_cursor[pos] - w * scale_r
        left_cursor[label], right_cursor[pos] = b0, b1
        ax.add_patch(patches.PathPatch(
            _flow_path(x_left + node_w, x_right - node_w, t0, b0, t1, b1),
            facecolor=colors[label], alpha=0.55, edgecolor="none"))

    # A label on a thick ribbon needs a chip behind it to stay readable; a label on a
    # hairline ribbon reads fine on white and a chip would only add clutter. The
    # threshold is "band is at least as tall as the text", so it tracks figure height
    # instead of drifting as the concept count changes the aspect ratio.
    if chip_min_height is None:
        chip_min_height = fontsize * 1.15 / (ax.figure.get_size_inches()[1] * 72) * 1.07

    def _chip(height):
        if height < chip_min_height:
            return None
        return dict(boxstyle="round,pad=0.22", facecolor=LABEL_CHIP, edgecolor="none")

    for label in concept_labels:
        top, bot = left[label]
        ax.add_patch(patches.Rectangle((x_left, bot), node_w, top - bot,
                                       facecolor=colors[label], edgecolor="none"))
        ax.text(x_left + node_w + pad, (top + bot) / 2, label, ha="left", va="center",
                fontsize=fontsize, color="#1a1a1a", bbox=_chip(top - bot), zorder=3)

    for pos, class_name in enumerate(classes):
        top, bot = right[pos]
        ax.add_patch(patches.Rectangle((x_right - node_w, bot), node_w, top - bot,
                                       facecolor=class_color[pos], edgecolor="none"))
        ax.text(x_right - node_w - pad, (top + bot) / 2, class_name, ha="right",
                va="center", fontsize=fontsize, color="#1a1a1a",
                bbox=_chip(top - bot), zorder=3)

    if headers:
        ax.text(x_left, 1.015, "Concept", ha="left", va="bottom",
                fontsize=fontsize + 2, color="#333333")
        ax.text(x_right, 1.015, "Prediction", ha="right", va="bottom",
                fontsize=fontsize + 2, color="#333333")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.07)
    ax.axis("off")
    ax.set_title(_sankey_title(run, classes) if title is None else title,
                 fontsize=fontsize + 4, pad=16)
    return concept_labels


def sankey_static(run, classes, weight_cutoff=0.05, max_per_class=12,
                  save_path=None, figsize=None, gap=0.012, class_colors=None,
                  concept_colors=None, dominance=0.85, fontsize=11, title=None,
                  headers=True):
    """Concept -> class flow diagram in the style of the paper's Figure 3.

    Same content as sankey() with no plotly dependency, and a static figure that drops
    straight into a write-up. Ribbon width is |w| from the sparse final layer, negative
    weights are labelled "NOT <concept>", and each ribbon takes the colour of its source
    concept -- all following the paper, which drew these by hand in sankeymatic.com.

    Colour marks *which class a concept decides*, not the sign of the weight: a concept
    that feeds one class takes that class's colour, one that feeds both stays grey. The
    paper instead groups colours by semantic similarity, edited by hand; pass
    `concept_colors={"citrus fruit": "#5b8fd4", ...}` to do the same here.

    Args:
        classes: the two (or more) class names to contrast, e.g. ["Felidae", "Canidae"].
        max_per_class: keep only this many largest-|w| concepts per class. The paper's
            figures show ~10-15 -- past that the labels stop being readable.
        dominance: share of a concept's flow that must go to one class before it takes
            that class's colour rather than grey.
    """
    import matplotlib.pyplot as plt

    classes = _as_class_list(run, classes)
    n_rows = len({e[0] for e in _sankey_edges(run, classes, weight_cutoff, max_per_class)})
    if figsize is None:
        figsize = (11, max(4.5, 0.45 * n_rows))

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    drawn = _draw_sankey(ax, run, classes, weight_cutoff=weight_cutoff,
                         max_per_class=max_per_class, gap=gap,
                         class_colors=class_colors, concept_colors=concept_colors,
                         dominance=dominance, fontsize=fontsize, title=title,
                         headers=headers)
    if not drawn:
        plt.close(fig)
        return None

    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path, dpi=200)
    plt.show()
    return fig


def sankey_panels(run, pairs, save_path=None, figsize=None, panel_width=9.0, **kwargs):
    """The paper's Figure 3 layout: one panel per class pair, side by side.

        ca.sankey_panels(run, [("Felidae", "Canidae"), ("Ursidae", "Mustelidae")])

    Every panel is scaled independently, so ribbon widths are comparable within a panel
    but not across panels -- as in the paper, where the two datasets share no scale.
    """
    import matplotlib.pyplot as plt

    pairs = [_as_class_list(run, p) for p in pairs]
    max_rows = max(len({e[0] for e in _sankey_edges(run, p,
                                                    kwargs.get("weight_cutoff", 0.05),
                                                    kwargs.get("max_per_class", 12))})
                   for p in pairs)
    if figsize is None:
        figsize = (panel_width * len(pairs), max(4.5, 0.45 * max_rows))

    fig, axes = plt.subplots(1, len(pairs), figsize=figsize, facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, classes in zip(axes, pairs):
        _draw_sankey(ax, run, classes, **kwargs)

    plt.tight_layout(w_pad=3.0)
    if save_path:
        _save_fig(fig, save_path, dpi=200)
    plt.show()
    return fig


# ------------------------------------------------------------------- collages

def _collage_image(run, raw_data, proc_data, idx, size=224):
    """Undecoded image when available -- avoids showing the normalised, cropped tensor."""
    try:
        return np.asarray(raw_data[idx][0].convert("RGB").resize((size, size))) / 255.0
    except Exception:
        if proc_data is None or run is None:
            raise
        return _to_displayable(proc_data[idx][0], run.preprocess)


def select_examples(results, mode="random", n=10, seed=0):
    """Indices for one collage. Modes: random, confident_correct, confident_wrong."""
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(len(results.labels), generator=g)[:n].tolist()

    conf = results.confidence
    if mode == "confident_correct":
        pool = results.correct.nonzero(as_tuple=True)[0]
    elif mode == "confident_wrong":
        pool = (~results.correct).nonzero(as_tuple=True)[0]
    else:
        raise ValueError("mode must be random, confident_correct or confident_wrong")

    if len(pool) == 0:
        return []
    order = torch.argsort(conf[pool], descending=True)
    return pool[order][:n].tolist()


def result_collage(run, results, mode="random", n=10, seed=0, ncols=5,
                   save_path=None, title=None):
    """Grid of example predictions, captioned with truth, prediction and confidence.

    Green frame = correct, red = wrong, so a "confident_wrong" sheet reads as failures
    at a glance without checking every caption.
    """
    import matplotlib.pyplot as plt

    chosen = select_examples(results, mode=mode, n=n, seed=seed)
    if not chosen:
        print("no examples for mode={}".format(mode))
        return None

    raw_data = get_data(run, results.split, raw=True)
    proc_data = get_data(run, results.split)
    conf = results.confidence

    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(chosen):]:
        ax.axis("off")

    for ax, idx in zip(axes, chosen):
        true_i, pred_i = int(results.labels[idx]), int(results.preds[idx])
        ok = true_i == pred_i
        ax.imshow(_collage_image(run, raw_data, proc_data, idx))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ca02c" if ok else "#d62728")
            spine.set_linewidth(3)
        caption = "#{}  p={:.2f}\ntrue: {}".format(idx, conf[idx].item(), run.classes[true_i])
        if not ok:
            caption += "\npred: {}".format(run.classes[pred_i])
        ax.set_title(caption, fontsize=7.5, color="#2ca02c" if ok else "#d62728")

    fig.suptitle(title or "{} -- {} ({})".format(run.name, mode.replace("_", " "), results.split),
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return fig


def save_collages(run, results, out_dir=None, n=10, seed=0):
    """All three collages -- random, confidently correct, confidently wrong -- to disk."""
    out_dir = out_dir or os.path.join(run.load_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for mode in ("random", "confident_correct", "confident_wrong"):
        path = os.path.join(out_dir, "collage_{}.png".format(mode))
        result_collage(run, results, mode=mode, n=n, seed=seed, save_path=path)
        paths.append(path)
    return paths


# ------------------------------------------------- per-class accuracy / confusion

def per_class_accuracy(run, results):
    """Per-class accuracy rows, worst first: class, accuracy, correct, support."""
    labels, preds = results.labels.numpy(), results.preds.numpy()
    rows = []
    for ci, name in enumerate(run.classes):
        mask = labels == ci
        support = int(mask.sum())
        if support == 0:
            continue
        n_correct = int((preds[mask] == ci).sum())
        rows.append({"class": name, "class_idx": ci, "accuracy": n_correct / support,
                     "correct": n_correct, "support": support})
    rows.sort(key=lambda r: (r["accuracy"], -r["support"]))
    return rows


def _support_warning(rows):
    """Birds525 ships ~5 val images per class, which makes per-class accuracy coarse."""
    supports = sorted(r["support"] for r in rows)
    median = supports[len(supports) // 2]
    if median < 10:
        print("NOTE: median support is {} images/class -- per-class accuracy can only take "
              "{} distinct values, so 'worst classes' is noisy. Treat as a shortlist to "
              "inspect, not a ranking.".format(median, median + 1))
    return median


def plot_class_accuracy(run, results, worst_k=25, save_path=None):
    """Distribution of per-class accuracy, plus the worst-k classes by name."""
    import matplotlib.pyplot as plt

    rows = per_class_accuracy(run, results)
    _support_warning(rows)
    worst = rows[:worst_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(5, 0.32 * worst_k)),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    accs = [r["accuracy"] for r in rows]
    ax1.hist(accs, bins=20, color="#4c72b0", edgecolor="white")
    ax1.axvline(results.accuracy, color="#d62728", ls="--",
                label="overall {:.1f}%".format(results.accuracy * 100))
    ax1.set_xlabel("per-class accuracy"); ax1.set_ylabel("classes")
    ax1.set_title("distribution over {} classes".format(len(rows)))
    ax1.legend(fontsize=8)

    ypos = range(len(worst))[::-1]
    ax2.barh(list(ypos), [r["accuracy"] for r in worst], color="#d62728")
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(["{} ({}/{})".format(r["class"], r["correct"], r["support"])
                         for r in worst], fontsize=7.5)
    ax2.set_xlabel("accuracy"); ax2.set_xlim(0, 1)
    ax2.set_title("worst {} classes".format(len(worst)))

    fig.suptitle("{} -- per-class accuracy ({})".format(run.name, results.split), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return rows


def best_worst_classes(run, results, k=5, min_support=1):
    """(best, worst) class names by accuracy -- what to point the Sankey at."""
    rows = [r for r in per_class_accuracy(run, results) if r["support"] >= min_support]
    rows.sort(key=lambda r: r["accuracy"])
    return [r["class"] for r in rows[-k:][::-1]], [r["class"] for r in rows[:k]]


def class_table(run, results, sort="accuracy", n=25, ascending=True):
    """Per-class accuracy with support, for choosing a class to inspect.

    Prints correct/support alongside accuracy because the accuracy alone is coarse:
    birds525's val split has 5 images per class, so it takes only 6 distinct values and
    "best" and "worst" are large tie groups rather than a ranking. Sort by "name" to
    scan alphabetically instead.
    """
    rows = per_class_accuracy(run, results)
    supports = {r["support"] for r in rows}
    if sort == "name":
        rows.sort(key=lambda r: r["class"])
    else:
        rows.sort(key=lambda r: (r["accuracy"], r["class"]), reverse=not ascending)

    shown = rows if n is None else rows[:n]
    print(f"{'class':38s} {'correct':>9s} {'accuracy':>9s}")
    for r in shown:
        print(f"  {r['class']:36s} {r['correct']:3d}/{r['support']:<3d} {r['accuracy'] * 100:8.1f}%")
    if len(supports) == 1:
        k = supports.pop()
        ties = sum(1 for r in rows if r["accuracy"] == shown[0]["accuracy"]) if shown else 0
        print(f"\n{len(rows)} classes, {k} images each -> accuracy takes only {k + 1} values; "
              f"{ties} classes tie at {shown[0]['accuracy'] * 100:.0f}%" if shown else "")
    return rows


def confused_pairs(run, results, k=20):
    """Most frequent (true -> predicted) mistakes, as a list of dicts."""
    import collections
    counts = collections.Counter(
        (int(t), int(p)) for t, p in zip(results.labels, results.preds) if t != p)
    return [{"true": run.classes[t], "pred": run.classes[p], "count": n}
            for (t, p), n in counts.most_common(k)]


def plot_confusion(run, results, worst_k=25, save_path=None, annotate=True):
    """Confusion submatrix over the worst-k classes and whatever they get mistaken for.

    The full matrix is 525x525 and unreadable, and with ~5 images per class it is
    almost entirely zeros -- so this restricts rows to the classes that actually fail
    and columns to the predictions they actually receive.
    """
    import matplotlib.pyplot as plt

    rows = per_class_accuracy(run, results)
    _support_warning(rows)
    worst = rows[:worst_k]
    row_idx = [r["class_idx"] for r in worst]

    labels, preds = results.labels.numpy(), results.preds.numpy()
    col_counts = {}
    for ci in row_idx:
        for p in preds[labels == ci]:
            col_counts[int(p)] = col_counts.get(int(p), 0) + 1
    keep = sorted(col_counts, key=lambda c: -col_counts[c])[:worst_k + 10]
    col_idx = sorted(set(keep) | set(row_idx), key=lambda c: run.classes[c])
    col_pos = {c: j for j, c in enumerate(col_idx)}

    # capping the columns can exclude a rare prediction one of these rows actually made;
    # bucket those into a trailing column so every row still sums to its support
    spill = sum(1 for ci in row_idx for p in preds[labels == ci] if int(p) not in col_pos)
    mat = np.zeros((len(row_idx), len(col_idx) + (1 if spill else 0)), dtype=int)
    for i, ci in enumerate(row_idx):
        for p in preds[labels == ci]:
            mat[i, col_pos.get(int(p), len(col_idx))] += 1

    col_names = [run.classes[c] for c in col_idx] + (["(other)"] if spill else [])
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(col_names)), max(6, 0.36 * len(row_idx))))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")
    fig.colorbar(im, ax=ax, label="images", fraction=0.025)

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_idx)))
    ax.set_yticklabels(["{} ({}/{})".format(r["class"], r["correct"], r["support"])
                        for r in worst], fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (worst classes)")

    # mark the diagonal so correct predictions are distinguishable from confusions
    for i, ci in enumerate(row_idx):
        if ci in col_pos:
            ax.add_patch(plt.Rectangle((col_pos[ci] - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#2ca02c", lw=1.6))
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=6.5,
                            color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    ax.set_title("{} -- confusions for the {} weakest classes ({})\n"
                 "green box = correct cell".format(run.name, len(row_idx), results.split),
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_fig(fig, save_path, dpi=150)
    plt.show()
    return mat


# ------------------------------------------------- cross-model qualitative figure

class ModelOutputs:
    """One run's predictions, retained after the model itself has been released.

    evaluate_many() builds these so several runs can be compared without five
    backbones resident on the GPU at once.
    """

    def __init__(self, name, backbone, dataset, classes, concepts, final_weight,
                 preds, labels, logits, concept_acts=None, final_bias=None):
        self.name = name
        self.backbone = backbone
        self.dataset = dataset
        self.classes = classes
        self.concepts = concepts
        self.final_weight = final_weight
        self.final_bias = final_bias
        self.preds = preds
        self.labels = labels
        self.logits = logits
        self.concept_acts = concept_acts

    @property
    def accuracy(self):
        return (self.preds == self.labels).float().mean().item()

    @property
    def correct(self):
        return self.preds == self.labels

    @property
    def confidence(self):
        return torch.softmax(self.logits, dim=1).max(dim=1).values

    def top_concepts(self, idx, k=2):
        """Highest-magnitude concept contributions for one example's prediction."""
        if self.concept_acts is None:
            return []
        weights = self.final_weight[int(self.preds[idx])]
        contrib = self.concept_acts[idx] * weights
        order = torch.argsort(contrib.abs(), descending=True)[:k]
        return [("" if contrib[i] >= 0 else "NOT ") + self.concepts[i] for i in order]

    def __repr__(self):
        return "<ModelOutputs {} acc={:.4f}>".format(self.name, self.accuracy)


def evaluate_many(load_dirs, split="val", device=None, batch_size=256,
                  keep_concept_acts=False):
    """Evaluate several runs in turn, freeing each backbone before loading the next.

    Five ViT-B/16 backbones will not fit comfortably on an 11GB card together, so only
    predictions and the final-layer weights are kept. Pass keep_concept_acts=True if you
    want per-example concept attributions (~20MB per run on birds525).
    """
    outs = []
    for d in load_dirs:
        run = load_run(d, device)
        res = evaluate(run, split=split, batch_size=batch_size)
        outs.append(ModelOutputs(
            run.name, run.backbone, run.dataset, run.classes, run.concepts,
            run.model.final.weight.detach().cpu(),
            res.preds, res.labels, res.logits,
            res.concept_acts if keep_concept_acts else None,
            run.model.final.bias.detach().cpu()))
        print("  {:42s} acc {:.2f}%".format(run.name, outs[-1].accuracy * 100))
        del res, run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outs


def _pick(outs, name):
    for o in outs:
        if o.name == name or o.backbone == name:
            return o
    raise ValueError("{!r} matched no run; have {}".format(
        name, [o.name for o in outs]))


def explanation_concentration(out, idx, k=2):
    """Share of the prediction's total concept contribution captured by the top-k shown.

    A low value means the concepts the figure displays are a small slice of what actually
    moved the logit -- the caption is not describing the decision, even when it is correct.
    """
    if out.concept_acts is None:
        return None
    contrib = out.concept_acts[idx] * out.final_weight[int(out.preds[idx])]
    total = contrib.abs().sum()
    if total == 0:
        return 0.0
    return float(contrib.abs().topk(min(k, contrib.numel())).values.sum() / total)


def explanation_sufficient(out, idx, k=2):
    """Does keeping ONLY the top-k displayed concepts still give the same prediction?

    Exact, not approximate: the head is linear, so masking the activation vector and
    recomputing every class logit is arithmetic on stored tensors. False means the shown
    concepts do not account for the decision -- a right answer for undisplayed reasons.
    """
    if out.concept_acts is None:
        return None
    act = out.concept_acts[idx]
    pred = int(out.preds[idx])
    keep = (act * out.final_weight[pred]).abs().topk(min(k, act.numel())).indices
    masked = torch.zeros_like(act)
    masked[keep] = act[keep]
    logits = masked @ out.final_weight.T
    if out.final_bias is not None:
        logits = logits + out.final_bias
    return int(logits.argmax()) == pred


def flag_explanation(out, idx, k=2, mode="sufficiency", threshold=0.25):
    """True when the displayed explanation should be treated as unreliable.

    No dataset here carries per-image trait labels, so this cannot detect a biologically
    *wrong* concept -- only an *unfaithful* one. Treat it as a shortlist for manual review,
    and override with story_figure(manual_wrong=...) when you have judged an example.
    """
    if mode in (None, "none") or out.concept_acts is None:
        return False
    if mode == "sufficiency":
        return explanation_sufficient(out, idx, k) is False
    if mode == "concentration":
        c = explanation_concentration(out, idx, k)
        return c is not None and c < threshold
    raise ValueError("mode must be sufficiency, concentration or none")


def story_examples(outs, strong=None, domain=None):
    """One representative index per narrative mode, plus how many images qualify.

    Candidates are ranked so the chosen image makes the point vividly -- for the
    "wins" modes that means both models were confident, not marginal.
    """
    ranked = sorted(outs, key=lambda o: -o.accuracy)
    domain_o = (_pick(outs, domain) if domain else
                next((o for o in outs if o.backbone == "bioclip"), ranked[-1]))
    # strong must differ from domain, or both "wins" rows are empty by construction
    # -- which happens whenever the domain model is also the most accurate one
    strong_o = (_pick(outs, strong) if strong else
                next(o for o in ranked if o is not domain_o))
    if strong_o is domain_o:
        raise ValueError("strong and domain resolved to the same run ({}); "
                         "pass them explicitly".format(strong_o.name))

    preds = torch.stack([o.preds for o in outs])
    correct = torch.stack([o.correct for o in outs])
    conf = torch.stack([o.confidence for o in outs])
    n = preds.shape[1]

    n_distinct = torch.tensor([len(set(preds[:, i].tolist())) for i in range(n)])
    s_ok, d_ok = strong_o.correct, domain_o.correct
    s_conf, d_conf = strong_o.confidence, domain_o.confidence

    modes = {
        "max_disagreement": (n_distinct >= 2, n_distinct.float() + conf.mean(0)),
        "strong_wins": (s_ok & ~d_ok, s_conf + d_conf),
        "domain_wins": (d_ok & ~s_ok, s_conf + d_conf),
        "unanimous_failure": (~correct.any(0), conf.mean(0)),
    }

    out = {}
    for mode, (mask, score) in modes.items():
        pool = mask.nonzero(as_tuple=True)[0]
        if len(pool) == 0:
            out[mode] = {"index": None, "count": 0}
        else:
            best = pool[torch.argmax(score[pool])].item()
            out[mode] = {"index": best, "count": int(len(pool))}
    out["_strong"], out["_domain"] = strong_o.name, domain_o.name
    return out


def story_figure(outs, split="val", strong=None, domain=None, top_concepts=0,
                 save_path=None, run_for_images=None, wrap=16,
                 flag_mode="sufficiency", flag_k=None, flag_threshold=0.25,
                 manual_wrong=None, flag_fn=None):
    """Single figure: four narrative rows, each one image judged by every model.

    Rows are max disagreement, the strong model winning, the domain model winning, and
    everything failing. Both "wins" rows are shown deliberately -- a figure containing
    only the cases your preferred model wins reads as cherry-picking.

    Cells are three-state: green for a correct prediction whose displayed concepts hold
    up, amber ("OK*") for a correct prediction whose explanation does not, red for a wrong
    prediction. No dataset here has per-image trait labels, so the amber test is
    faithfulness, not biological correctness -- pass manual_wrong=[(run_name, idx), ...]
    to override it with your own judgement, which is what a paper figure should rest on.
    Set flag_mode=None to disable flagging entirely.
    """
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    picks = story_examples(outs, strong=strong, domain=domain)
    strong_name, domain_name = picks.pop("_strong"), picks.pop("_domain")

    order = ["max_disagreement", "strong_wins", "domain_wins", "unanimous_failure"]
    titles = {
        "max_disagreement": "models disagree most",
        "strong_wins": "{} right, {} wrong".format(_short(strong_name), _short(domain_name)),
        "domain_wins": "{} right, {} wrong".format(_short(domain_name), _short(strong_name)),
        "unanimous_failure": "every model wrong",
    }
    rows = [m for m in order if picks[m]["index"] is not None]
    if not rows:
        print("no qualifying examples in any mode")
        return None

    if run_for_images is not None:
        raw_data = get_data(run_for_images, split, raw=True)
        proc_data = get_data(run_for_images, split)
    else:
        import torchvision.transforms as T
        raw_data = data_utils.get_data("{}_{}".format(outs[0].dataset, split),
                                       preprocess=T.Lambda(lambda x: x))
        proc_data = None

    manual = set(tuple(m) for m in (manual_wrong or []))
    k_flag = flag_k or (top_concepts or 2)
    flagged_any = False

    n_col = len(outs) + 1
    fig = plt.figure(figsize=(2.05 * n_col + 1.6, 2.75 * len(rows)))
    gs = gridspec.GridSpec(len(rows), n_col, figure=fig,
                           width_ratios=[1.25] + [1] * len(outs),
                           hspace=0.32, wspace=0.12)

    for r, mode in enumerate(rows):
        idx = picks[mode]["index"]
        truth = int(outs[0].labels[idx])

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(_collage_image(run_for_images, raw_data, proc_data, idx))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#333333"); s.set_linewidth(1.2)
        ax.set_ylabel("{}\n({} imgs)".format(titles[mode], picks[mode]["count"]),
                      fontsize=8.5, fontweight="bold", labelpad=8)
        ax.set_title("#{}  truth:\n{}".format(
            idx, "\n".join(textwrap.wrap(outs[0].classes[truth], wrap))), fontsize=7.5)

        for c, o in enumerate(outs):
            cell = fig.add_subplot(gs[r, c + 1])
            cell.set_xticks([]); cell.set_yticks([])
            ok = bool(o.correct[idx])

            # precedence: a judgement you made by hand, then a flag_fn (e.g. a VLM
            # actually looking at the image), then the faithfulness proxy
            if (o.name, idx) in manual:
                suspect = True
            elif flag_fn is not None:
                suspect = bool(flag_fn(o, idx, ok))
            elif ok:
                suspect = flag_explanation(o, idx, k=k_flag, mode=flag_mode,
                                           threshold=flag_threshold)
            else:
                suspect = False
            if suspect:
                flagged_any = True

            if not ok:
                mark, colour, fill = "X", "#d62728", "#ffebee"
            elif suspect:
                mark, colour, fill = "OK*", "#c77c00", "#fff8e1"
            else:
                mark, colour, fill = "OK", "#2ca02c", "#e8f5e9"

            cell.set_facecolor(fill)
            for s in cell.spines.values():
                s.set_edgecolor(colour); s.set_linewidth(1.8)

            if r == 0:
                cell.set_title("{}\n{:.1f}%".format(_short(o.name), o.accuracy * 100),
                               fontsize=8.5, fontweight="bold")

            body = "\n".join(textwrap.wrap(o.classes[int(o.preds[idx])], wrap))
            cell.text(0.5, 0.80, mark, ha="center", va="top",
                      fontsize=11, fontweight="bold",
                      color=colour, transform=cell.transAxes)
            cell.text(0.5, 0.62, body, ha="center", va="top", fontsize=7.2,
                      transform=cell.transAxes)
            cell.text(0.5, 0.06, "p={:.2f}".format(o.confidence[idx]), ha="center",
                      va="bottom", fontsize=7, color="#555555", transform=cell.transAxes)
            if top_concepts:
                cs = o.top_concepts(idx, k=top_concepts)
                if cs:
                    cell.text(0.5, 0.20, "\n".join(textwrap.wrap(", ".join(cs), 22)),
                              ha="center", va="bottom", fontsize=5.8, style="italic",
                              color="#444444", transform=cell.transAxes)

    title = "Qualitative comparison on {} -- one image per failure mode".format(split)
    if flagged_any:
        how = ("VLM judge" if flag_fn is not None
               else "manual" if flag_mode in (None, "none") else flag_mode)
        title += ("\nOK* = correct prediction, displayed concepts do not account for it"
                  " ({}, k={})".format(how, k_flag))
    fig.suptitle(title, fontsize=12, y=0.995)
    if save_path:
        _save_fig(fig, save_path, dpi=200)
    plt.show()
    return fig


def _short(name):
    """Compact run name for column headers."""
    return (name.replace("_birds525", "").replace("birds525_", "")
                .replace("__lam", " lam").replace("_vitb_concepts", "+vitbC"))


# ---------------------------------------------------------------------------
# VLG-CBM-only analyses. These need the Grounding DINO annotations, so they have
# no Label-free-CBM counterpart -- LF-CBM never sees where a concept is.
# ---------------------------------------------------------------------------


def explain_with_boxes(run, idx, split="val", top_k=8, annotation_dir=None, figsize=(15, 5)):
    """The per-decision bar plot, next to the image with the detector's boxes drawn on.

    This is the question LF-CBM cannot ask: the bars say which concepts drove the
    prediction, and the boxes say whether those concepts were ever actually found in
    the image. A concept carrying a large positive contribution with no box is being
    used as a class prior, not as evidence -- which is exactly the failure mode a
    non-localisable concept produces.

    Boxes are drawn only for concepts in the displayed top-k, and the legend marks
    which of those had no detection at all.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    proc = get_data(run, split)
    raw = get_data(run, split, raw=True)
    image, label = proc[idx]
    pil_image = raw[idx][0]

    with torch.no_grad():
        logits, concept_act = run.model(image.unsqueeze(0).to(run.device))
    pred = int(logits.argmax(dim=1))
    labels, contrib = _top_contributions(run, concept_act[0], pred, top_k=top_k)

    shown = [l.replace("NOT ", "") for l in labels]
    try:
        boxes = annotations_for(run, idx, split, annotation_dir)
    except FileNotFoundError as e:
        print("note: {}; drawing the image without boxes".format(e))
        boxes = []
    detected = {b["label"] for b in boxes}

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [1, 1.3]})
    axes[0].imshow(pil_image)
    axes[0].axis("off")

    palette = plt.get_cmap("tab10")
    colors = {c: palette(i % 10) for i, c in enumerate(shown)}
    drawn = set()
    for b in boxes:
        if b["label"] not in colors:
            continue
        x0, y0, x1, y1 = b["box"]
        axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=colors[b["label"]], lw=2))
        if b["label"] not in drawn:
            axes[0].text(x0, max(y0 - 4, 8), b["label"], fontsize=7, color="white",
                         bbox=dict(facecolor=colors[b["label"]], edgecolor="none",
                                   pad=1, alpha=0.85))
            drawn.add(b["label"])

    ungrounded = [c for c in shown if c not in detected]
    title = "true: {}\npred: {}".format(run.classes[int(label)], run.classes[pred])
    if ungrounded:
        title += "\nno box for {} of top {}".format(len(ungrounded), len(shown))
    axes[0].set_title(title, fontsize=9,
                      color="black" if pred == int(label) else "firebrick")

    # Same order and colours as explain_example: _top_contributions already ranks by
    # |contribution|, and re-sorting by signed value here made the two plots disagree on
    # the tail (a -0.46 outranks a +0.44 by magnitude but not by value) while red/blue
    # also meant the opposite thing in each. Same numbers, so the only thing that
    # differed was which plot you happened to be reading.
    order = np.arange(len(contrib))
    y = np.arange(len(order))[::-1]
    bar_colors = ["tab:red" if contrib[i] > 0 else "tab:blue" for i in order]
    axes[1].barh(y, contrib[order], color=bar_colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        ["{}{}".format(labels[i], "" if shown[i] in detected else "  (no box)")
         for i in order], fontsize=8)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("contribution to predicted class logit")
    axes[1].set_title("top {} concepts".format(len(order)), fontsize=9)

    fig.tight_layout()
    return fig


def concept_agreement(run, results=None, split="val", annotation_dir=None,
                      batch_size=256, num_workers=4, min_support=5):
    """Per-concept agreement between what the CBL predicts and what the detector found.

    The CBL is trained to reproduce the annotation labels, so this measures how well it
    learned that job on held-out data -- and, read the other way, which concepts are not
    learnable from the image at all.

    Returns rows sorted by AUC ascending, so the least grounded concepts come first.
    A concept near AUC 0.5 is one the backbone cannot distinguish, which for the
    birds525 set tends to mean the abstract ones ("birdsong", "oscine") that Grounding
    DINO grounded to something arbitrary. `support` is how many images carried it.
    """
    if results is None:
        results = evaluate(run, split=split, batch_size=batch_size, num_workers=num_workers)

    n = len(results.labels)
    adir = annotation_dir_for(run, split, annotation_dir)
    threshold = run.train_args.get("cbl_confidence_threshold", 0.15)
    index = {c: i for i, c in enumerate(run.concepts)}

    truth = torch.zeros(n, len(run.concepts))
    for i in range(n):
        path = os.path.join(adir, "{}.json".format(i))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "no annotation at {} -- concept_agreement needs the annotations for "
                "this split".format(path))
        with open(path) as f:
            data = json.load(f)
        for a in data[1:]:
            if a["logit"] > threshold and a["label"] in index:
                truth[i, index[a["label"]]] = 1.0

    acts = results.concept_acts
    rows = []
    for j, concept in enumerate(run.concepts):
        y = truth[:, j]
        pos = int(y.sum().item())
        if pos < min_support or pos == n:
            continue
        rows.append({"concept": concept, "support": pos,
                     "auc": _auc(acts[:, j], y),
                     "mean_act_pos": float(acts[y == 1, j].mean()),
                     "mean_act_neg": float(acts[y == 0, j].mean())})

    rows.sort(key=lambda r: r["auc"])
    skipped = len(run.concepts) - len(rows)
    if skipped:
        print("note: {} of {} concepts scored on fewer than {} positive examples "
              "and were skipped".format(skipped, len(run.concepts), min_support))
    return rows


def _auc(scores, y):
    """ROC AUC via the rank identity, so scipy/sklearn are not needed."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def plot_concept_agreement(run, rows=None, results=None, split="val", k=20,
                           save_path=None, **kwargs):
    """The worst- and best-grounded concepts, side by side."""
    import matplotlib.pyplot as plt

    if rows is None:
        rows = concept_agreement(run, results=results, split=split, **kwargs)
    if not rows:
        print("no concepts had enough support to score")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, k * 0.28)))
    for ax, subset, title in (
            (axes[0], rows[:k], "least grounded (AUC closest to chance)"),
            (axes[1], rows[-k:][::-1], "best grounded")):
        y = np.arange(len(subset))
        ax.barh(y, [r["auc"] for r in subset],
                color=["firebrick" if r["auc"] < 0.6 else "tab:blue" for r in subset])
        ax.set_yticks(y)
        ax.set_yticklabels(["{}  (n={})".format(r["concept"], r["support"]) for r in subset],
                           fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.5, color="black", lw=0.8, ls="--")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUC: CBL activation vs Grounding DINO label")
        ax.set_title("{} -- {}".format(run.name, title), fontsize=9)

    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# VLM judge. story_figure's amber cells are set by a proxy (explanation_sufficient)
# that can only detect an *unfaithful* explanation -- one whose displayed concepts do
# not account for the logit -- never a *wrong* one, because no per-image trait labels
# exist. A VLM that looks at the image and answers "is this concept visible?" tests the
# thing the proxy cannot: whether the concept the model leveraged is actually there.
# ---------------------------------------------------------------------------


class VLMJudge:
    """Asks a small VLM whether a concept is visible in an image.

    Scores by comparing the logits of " Yes" and " No" for the first generated token
    rather than parsing generated text, so the answer is deterministic, needs one forward
    pass, and yields a probability that can be thresholded instead of a bare boolean.

    Defaults to Qwen2-VL-2B in float16: ~4.4GB, and T4 is Turing, which has no bfloat16.
    """

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", device="cuda", dtype=None):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if dtype is None:
            # bf16 needs Ampere+; T4 and older silently fall back and run slowly
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.device, self.model_id = device, model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=dtype).to(device).eval()

        tok = self.processor.tokenizer
        # the leading space matters: chat templates put the answer after one
        self._yes = [i for s in (" Yes", "Yes") for i in tok.encode(s, add_special_tokens=False)[:1]]
        self._no = [i for s in (" No", "No") for i in tok.encode(s, add_special_tokens=False)[:1]]

    def score(self, image, concept):
        """P(yes) that `concept` is visible, in [0, 1]."""
        import torch

        prompt = (f'Look at this bird photograph. Is "{concept}" clearly visible in the '
                  f"image? Answer only Yes or No.")
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        yes = torch.logsumexp(logits[self._yes], 0)
        no = torch.logsumexp(logits[self._no], 0)
        return float(torch.softmax(torch.stack([no, yes]), 0)[1])

    def present(self, image, concept, threshold=0.5):
        return self.score(image, concept) >= threshold

    def sanity_check(self, run, results, split="val", n=6, seed=0, verbose=True):
        """Verify the judge discriminates before any of its answers are trusted.

        A judge that answers Yes to everything would silently mark every cell green and
        look like a clean result. So this compares two populations: concepts Grounding
        DINO actually detected in an image, and concepts drawn from a different image's
        detections. It returns the gap, and the caller should refuse to use a judge whose
        gap is not clearly positive.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        raw = get_data(run, split, raw=True)
        idxs = rng.choice(len(results.labels), size=n, replace=False)

        present_scores, absent_scores = [], []
        for i in idxs:
            try:
                here = {a["label"] for a in annotations_for(run, int(i), split)}
            except FileNotFoundError:
                continue
            j = int(rng.choice([k for k in idxs if k != i]))
            try:
                other = {a["label"] for a in annotations_for(run, j, split)} - here
            except FileNotFoundError:
                continue
            if not here or not other:
                continue
            img = raw[int(i)][0]
            present_scores.append(self.score(img, sorted(here)[rng.integers(len(here))]))
            absent_scores.append(self.score(img, sorted(other)[rng.integers(len(other))]))

        if not present_scores:
            raise RuntimeError("sanity check collected no pairs -- are annotations present?")
        p, a = float(np.mean(present_scores)), float(np.mean(absent_scores))
        if verbose:
            print(f"{self.model_id}")
            print(f"  mean P(yes) on DETECTED concepts : {p:.3f}  (n={len(present_scores)})")
            print(f"  mean P(yes) on OTHER-IMAGE concepts: {a:.3f}")
            print(f"  separation: {p - a:+.3f}")
            if p - a < 0.05:
                print("  WARNING: judge barely discriminates -- its flags would be noise. "
                      "Try a different model or rephrase the prompt before using it.")
            else:
                print("  OK: judge separates present from absent.")
        return {"present": p, "absent": a, "separation": p - a, "n": len(present_scores)}


def vlm_flag_fn(judge, run, split="val", k=2, threshold=0.5, cache=None, verbose=False):
    """Build a flag_fn for story_figure that asks the VLM instead of using the proxy.

    Flags a cell when a concept the model leveraged is not actually visible. A concept
    shown as "NOT x" means the model used the *absence* of x, so that one is flagged when
    x turns out to be present -- the contradiction is the mirror image.
    """
    raw = get_data(run, split, raw=True)
    cache = {} if cache is None else cache

    def flag(out, idx, ok):
        if not ok:
            return False
        concepts = out.top_concepts(idx, k=k)
        img = raw[int(idx)][0]
        for c in concepts:
            negated = c.startswith("NOT ")
            name = c[4:] if negated else c
            key = (int(idx), name)
            if key not in cache:
                cache[key] = judge.score(img, name)
            visible = cache[key] >= threshold
            if visible == negated:          # claimed present but absent, or vice versa
                if verbose:
                    print(f"  flag {out.name} idx={idx}: {c!r} score={cache[key]:.2f}")
                return True
        return False

    return flag
