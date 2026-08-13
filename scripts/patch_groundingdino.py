"""Make Grounding DINO run on GPU without its compiled CUDA extension.

The NRP `scientific-images/python` container has no CUDA toolkit (no nvcc, no
/usr/local/cuda), so `pip install -e GroundingDINO` silently builds without the
`groundingdino._C` extension -- setup.py only adds it when CUDA_HOME is set.

That leaves ms_deform_attn.py broken rather than degraded: the import of `_C` is
wrapped in try/except, but the forward pass still dispatches on
`torch.cuda.is_available() and value.is_cuda`, so any GPU tensor takes the branch
that calls the extension and dies with NameError. Grounding DINO already ships an
equivalent pure-PyTorch implementation (`multi_scale_deformable_attn_pytorch`,
grid_sample based, runs fine on GPU); this patch just re-points the dispatch at
whether the extension actually imported.

Idempotent -- running it twice is a no-op.

Usage:
    python scripts/patch_groundingdino.py --groundingdino_dir GroundingDINO
"""

import argparse
import os
import sys

TARGET = "groundingdino/models/GroundingDINO/ms_deform_attn.py"

IMPORT_BEFORE = """try:
    from groundingdino import _C
except:
    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
"""

IMPORT_AFTER = """try:
    from groundingdino import _C

    _C_AVAILABLE = True
except:
    _C_AVAILABLE = False
    warnings.warn("Failed to load custom C++ ops; using the pure-PyTorch deformable attention.")
"""

DISPATCH_BEFORE = "        if torch.cuda.is_available() and value.is_cuda:"
DISPATCH_AFTER = "        if _C_AVAILABLE and value.is_cuda:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundingdino_dir", type=str, default="GroundingDINO")
    args = parser.parse_args()

    path = os.path.join(args.groundingdino_dir, TARGET)
    with open(path) as f:
        source = f.read()

    if "_C_AVAILABLE" in source:
        print(f"{path} already patched")
        return

    for before in (IMPORT_BEFORE, DISPATCH_BEFORE):
        if before not in source:
            print(f"FATAL: could not find the following in {path}; upstream has changed:\n{before}")
            sys.exit(1)

    source = source.replace(IMPORT_BEFORE, IMPORT_AFTER)
    source = source.replace(DISPATCH_BEFORE, DISPATCH_AFTER)

    with open(path, "w") as f:
        f.write(source)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
