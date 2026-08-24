import json
import os
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
import torch
from torchvision import datasets, models, transforms
from tqdm import tqdm
from loguru import logger
import data.data_lp as data_lp
import clip
from PIL import Image

# get from the environment variable
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "datasets")

# BioCLIP is not on any hub we can reach from the cluster, so the checkpoint is
# staged on the PVC and located by env var (same convention as the LF-CBM repo).
BIOCLIP_CKPT = os.environ.get("VLGCBM_BIOCLIP_CKPT", "/workspace/models/bioclip/open_clip_pytorch_model.bin")
# BioCLIP 2 is published on the HF hub, so open_clip fetches it by name rather than from
# a staged file. Note it is ViT-L/14: 1024-d and ~3x the forward cost of BioCLIP 1.
BIOCLIP2_HUB = os.environ.get("VLGCBM_BIOCLIP2_HUB", "hf-hub:imageomics/bioclip-2")

DATASET_ROOTS = {
    "imagenet_train": f"{DATASET_FOLDER}/imagenet/ILSVRC/Data/CLS-LOC/train",
    "imagenet_val": f"{DATASET_FOLDER}/imagenet/ILSVRC/Data/CLS-LOC/ImageNet_val",
    "cub_train": f"{DATASET_FOLDER}/CUB/train",
    "cub_val": f"{DATASET_FOLDER}/CUB/test",
    # Datasets carried over from the LF-CBM experiments. `_val` is what VLG-CBM
    # calls the test set; the validation split is carved out of `_train` by
    # get_concept_dataloader using args.val_split.
    "birds525_train": f"{DATASET_FOLDER}/birds525/train",
    "birds525_val": f"{DATASET_FOLDER}/birds525/val",
    "treeoflife_train": f"{DATASET_FOLDER}/treeoflife/train",
    "treeoflife_val": f"{DATASET_FOLDER}/treeoflife/val",
    # First 5 classes of birds525, symlinked. Annotating and training on it takes
    # minutes, so it is the end-to-end check to run before the multi-hour birds525
    # jobs. See scripts/make_smoke_dataset.sh.
    "birds525mini_train": f"{DATASET_FOLDER}/birds525mini/train",
    "birds525mini_val": f"{DATASET_FOLDER}/birds525mini/val",
}

LABEL_FILES = {
    "places365": "concept_files/categories_places365_clean.txt",
    "imagenet": "concept_files/imagenet_classes.txt",
    "cifar10": "concept_files/cifar10_classes.txt",
    "cifar100": "concept_files/cifar100_classes.txt",
    "cub": "concept_files/cub_classes.txt",
    "food": "concept_files/food_classes.txt",
    "flower": "concept_files/flower_classes.txt",
    "aircraft": "concept_files/aircraft_classes.txt",
    "dtd": "concept_files/dtd_classes.txt",
    "birds525": "concept_files/birds525_classes.txt",
    "birds525mini": "concept_files/birds525mini_classes.txt",
    "treeoflife": "concept_files/treeoflife_classes.txt",
}

BACKBONE_ENCODING_DIMENSION = {
    "resnet18_cub": 512,
    "clip_RN50": 1024,
    "clip_RN50_penultimate": 2048,
    "resnet50": 2048,
    # BioCLIP is ViT-B/16: 768-d pre-projection features out of visual.ln_post,
    # not the 512-d output of encode_image.
    "bioclip": 768,
    # The rest of the LF-CBM backbone comparison. All ViT-B/16 except resnet50, so
    # capacity is held fixed and only pretraining differs.
    "vit_in21k": 768,
    "dino_vitb16": 768,
    # ViT-L/14, so wider than everything else here -- and not capacity-matched to it.
    "bioclip2": 1024,
    # CLIP ViT-B/16 emits the 512-d projected embedding. There is no _penultimate
    # entry because BackboneCLIP's penultimate path rewrites visual.attnpool, which
    # only exists on CLIP's ResNet towers -- use_clip_penultimate must stay false here.
    "clip_ViT-B/16": 512,
}

BACKBONE_VISUALIZATION_TARGET_LAYER = {
    "resnet18_cub": "features.stage4.unit2.body.conv2",
}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=False,
                transform=preprocess,
            )
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=False,
                transform=preprocess,
            )
    elif dataset_name == "food_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/food",
            split="train",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["food"],
        )
        data.targets = data.labels
    elif dataset_name == "food_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/food",
            split="test",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["food"],
        )
        data.targets = data.labels
    elif dataset_name == "dtd_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/dtd",
            split="train",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["dtd"],
        )
        data.targets = data.labels
    elif dataset_name == "dtd_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/dtd",
            split="test",
            transform=preprocess,
            img_ext="",
            cls_names_file=LABEL_FILES["dtd"],
        )
        data.targets = data.labels
    elif dataset_name == "flower_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/flower",
            split="train",
            transform=preprocess,
            cls_names_file=LABEL_FILES["flower"],
        )
        data.targets = data.labels
    elif dataset_name == "flower_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/flower",
            split="test",
            transform=preprocess,
            cls_names_file=LABEL_FILES["flower"],
        )
        data.targets = data.labels
    elif dataset_name == "aircraft_train":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/aircraft",
            split="train",
            transform=preprocess,
            cls_names_file=LABEL_FILES["aircraft"],
        )
        data.targets = data.labels
    elif dataset_name == "aircraft_val":
        data = data_lp.LinearProbeDataset(
            data_path=f"datasets/aircraft",
            split="test",
            transform=preprocess,
            cls_names_file=LABEL_FILES["aircraft"],
        )
        data.targets = data.labels
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset(
            [
                datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess),
                datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess),
            ]
        )
    return data


def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets


class BioCLIPBackbone(torch.nn.Module):
    """BioCLIP's image tower, tapped at visual.ln_post.

    Returns the 768-d pre-projection features instead of the 512-d output of
    encode_image, so the concept layer sees the true penultimate layer. Kept as
    a Module (not a lambda) so model.cbm.Backbone can resolve the
    `visual.ln_post` attribute path when registering its own hook.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x):
        # ln_post's output is what Backbone hooks; the return value only has to
        # live on the right device for Backbone to look the hook result up.
        return self.visual(x).float()


def load_bioclip(device):
    """Load the BioCLIP ViT-B/16 checkpoint into an open_clip model."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16")
    checkpoint = torch.load(BIOCLIP_CKPT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model, preprocess


# timm backbones used as baselines in the BioCLIP paper: same ViT-B/16 architecture as
# BioCLIP, different pretraining objective, so they isolate pretraining from scale.
TIMM_BACKBONES = {
    "vit_in21k": "vit_base_patch16_224.augreg_in21k",  # supervised ImageNet-21k
    "dino_vitb16": "vit_base_patch16_224.dino",        # DINO self-supervised
}


class TimmBackbone(torch.nn.Module):
    """A timm vision model exposed as a feature extractor.

    `self.out` is an Identity whose output is the final pooled feature, so hooking it
    (feature_layer "out") hands model.cbm.Backbone an already-2-D tensor. That keeps the
    pooling decision here rather than in the hook, which would otherwise mean-pool ViT
    tokens and silently discard the CLS token.
    """

    def __init__(self, model, pool="cls"):
        super().__init__()
        self.model = model
        self.pool = pool
        self.out = torch.nn.Identity()

    def forward(self, x):
        feats = self.model.forward_features(x)
        if feats.dim() == 3:  # (B, tokens, D)
            feats = feats[:, 0] if self.pool == "cls" else feats.mean(dim=1)
        elif feats.dim() == 4:  # (B, D, H, W)
            feats = feats.mean(dim=[2, 3])
        return self.out(feats.float())


def load_timm_backbone(target_name, device, pool="cls"):
    """Load a timm backbone plus the preprocessing that model was trained with.

    num_classes=0 drops the classifier head (in21k's is 21843-wide). The transform comes
    from the model's own data config -- augreg_in21k and DINO use different normalisation,
    so a shared ImageNet transform would be wrong for one of them.
    """
    import timm

    model = timm.create_model(TIMM_BACKBONES[target_name], pretrained=True, num_classes=0)
    model = model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**cfg)
    return TimmBackbone(model, pool=pool).to(device).eval(), preprocess


def load_bioclip2(device):
    """BioCLIP 2 (ViT-L/14) from the HF hub."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP2_HUB)
    return model, preprocess


def get_target_model(target_name, device):
    if target_name == "bioclip":
        model, preprocess = load_bioclip(device)
        target_model = BioCLIPBackbone(model).to(device).eval()

    elif target_name == "bioclip2":
        # Same wrapper as BioCLIP 1: model.cbm.Backbone hooks visual.ln_post and averages
        # the token sequence, which is width-agnostic.
        model, preprocess = load_bioclip2(device)
        target_model = BioCLIPBackbone(model).to(device).eval()

    elif target_name in TIMM_BACKBONES:
        target_model, preprocess = load_timm_backbone(target_name, device)

    elif target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name == "resnet18_places":
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load("data/resnet18_places365.pth.tar")["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name == "resnet18_cub":
        # Imported here rather than at module scope: pytorchcv pulls in the whole of
        # torchvision.models and torch._dynamo behind it, which measured 1.2s of the
        # 1.8s it took to import this module warm -- and far more cold, off the PVC.
        # resnet18_cub is the only thing that needs it.
        from pytorchcv.model_provider import get_model as ptcv_get_model

        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess


def format_concept(s):
    # replace - with ' '
    # replace , with ' '
    # only one space between words
    s = s.lower()
    s = s.replace("-", " ")
    s = s.replace(",", " ")
    s = s.replace(".", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    if s[:2] == "a ":
        s = s[2:]
    elif s[:3] == "an ":
        s = s[3:]

    # remove trailing and leading spaces
    s = " ".join(s.split())
    return s

def get_classes(dataset_name):
    with open(LABEL_FILES[dataset_name], "r") as f:
        classes = f.read().split("\n")
    return classes


def get_concepts(concept_file: str, filter_file:Optional[str]=None) -> List[str]:
    with open(concept_file) as f:
        concepts: List[str] = f.read().split("\n")

    # remove repeated concepts and maintain order
    concepts = list(dict.fromkeys([format_concept(concept) for concept in concepts]))

    # check for filter file
    if filter_file and os.path.exists(filter_file):
        logger.info(f"Filtering concepts using {filter_file}")
        with open(filter_file) as f:
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]

    return concepts


def save_concept_count(
    concepts: List[str],
    counts: List[int],
    save_dir: str,
    file_name: str = "concept_counts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(concepts) != len(counts):
            raise ValueError("Length of concepts and counts should be the same")
        f.write(f"{concepts[0]} {counts[0]}")
        for concept, count in zip(concepts[1:], counts[1:]):
            f.write(f"\n{concept} {count}")


def load_concept_and_count(
    save_dir: str, file_name: str = "concept_counts.txt", filter_file:Optional[str]=None
) -> Tuple[List[str], List[float]]:
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        concepts = []
        counts = []
        for line in lines:
            concept = line.split(" ")[:-1]
            concept = " ".join(concept)
            count = line.split(" ")[-1]
            concepts.append(format_concept(concept))
            counts.append(float(count))

    if filter_file and os.path.exists(filter_file):
        with open(filter_file) as f:
            logger.info(f"Filtering concepts using {filter_file}")
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        counts = [count for concept, count in zip(concepts, counts) if concept not in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]
        assert len(concepts) == len(counts)

    return concepts, counts

def save_filtered_concepts(
    filtered_concepts: List[str],
    save_dir: str,
    file_name: str = "filtered_concepts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(filtered_concepts) > 0:
            f.write(filtered_concepts[0])
            for concept in filtered_concepts[1:]:
                f.write("\n" + concept)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

def plot_annotations(image_pil: Image.Image, annotations: List[Dict]) -> plt.Figure:
    """
    Plot annotations on image

    Args:
        image_pil (Image.Image): The PIL image
        annotations (List[Dict]): The annotations to plot in the following format:
            - logits: The logits associated with each token of the concept.
            - score: The perplexity of the concept.
            - concept: The concept associated with the bounding box.
            - bbox: The bounding box coordinates.

    Returns:
        plt.Figure: The figure containing the image with annotations.
    """
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image_pil)
    for annotation in annotations:
        show_box(annotation["box"], plt.gca(), f"{annotation['label']} : {annotation['logit']:.3f}")
    plt.axis("off")
    return fig