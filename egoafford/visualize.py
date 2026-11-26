from pathlib import Path
import numpy as np
from PIL import Image
from .data import relative_file, load_masks


def save_overlay(row, root, output):
    with Image.open(relative_file(root, row['image'])) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=float)
    masks = load_masks(row, root)
    if rgb.shape[:2] != masks.shape[1:]:
        raise ValueError('Image/mask dimensions differ.')
    colors = np.asarray([[255, 0, 0], [0, 180, 0], [0, 110, 255]], dtype=float)
    count = masks.sum(axis=0)
    blend = np.einsum('rhw,rc->hwc', masks.astype(float), colors) / np.maximum(count[..., None], 1)
    rgb = np.where(count[..., None] > 0, rgb * .55 + blend * .45, rgb)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb).astype('uint8')).save(path)
