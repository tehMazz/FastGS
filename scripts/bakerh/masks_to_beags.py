"""SAM3 single-object id maps -> BEA-GS masks.

    python scripts/bakerh/masks_to_beags.py --mask_dir <.../masks_npy> --out <dir>

<mask_dir>/<stem>.npy is a 2D id map (0 = background, > 0 = object; (H, W, C) maps use channel 0).
<out>/<stem>.npy is the (2, H, W) uint8 mask BEA-GS reads: channel 0 background, channel 1 the object (label 1).
"""
import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mask_dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

files = sorted(args.mask_dir.glob("*.npy"))
assert files, f"no .npy masks in {args.mask_dir}"
args.out.mkdir(parents=True, exist_ok=True)
shapes, fractions = set(), []
for f in files:
    m = np.load(f)
    if m.ndim == 3:
        m = m[0] if m.shape[0] == 1 else m[..., 0]
    assert m.ndim == 2, f"{f}: expected a 2D id map, got shape {m.shape}"
    obj = m > 0
    np.save(args.out / f.name, np.stack([~obj, obj]).astype(np.uint8))
    shapes.add(obj.shape)
    fractions.append(obj.mean())
fractions = np.array(fractions)
print(f"{len(files)} masks -> {args.out}; shapes {sorted(shapes)}; object in {int((fractions > 0).sum())} views, "
      f"{100 * fractions.mean():.1f}% of the pixels on average")
