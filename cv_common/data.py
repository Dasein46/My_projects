"""Patient-level splitting and image integrity checks; no image-level fallback."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


def validate_images(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Resolve paths and reject repeated files/pixels before any split.

    Pixel hashes catch identical decoded images, not near duplicates or other
    views of the same patient. Patient metadata is still mandatory.
    """
    frame = frame.copy()
    if frame.empty or frame['filename'].isna().any():
        raise ValueError('Empty manifest or missing filename.')
    root = Path(root).resolve()
    paths, hashes = [], []
    for value in frame['filename']:
        path = (root / str(value)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Image must exist within the configured data root.')
        with Image.open(path) as img:
            if getattr(img, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images require an explicit frame policy.')
            # Do not silently clip medical 16-bit / floating point intensities.
            if img.mode not in {'RGB', 'RGBA', 'L', 'P', '1'}:
                raise ValueError(f'Explicit intensity conversion required for mode {img.mode}.')
            rgb = img.convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        paths.append(str(path))
        hashes.append(digest)
    if len(set(paths)) != len(paths) or len(set(hashes)) != len(hashes):
        raise ValueError('Repeated image/path detected; resolve duplicates and labels first.')
    frame['filename'], frame['pixel_sha256'] = paths, hashes
    return frame


def patient_split(frame: pd.DataFrame, seed: int = 12345) -> dict[str, pd.DataFrame]:
    """Use an existing split, or create 60/20/20 partitions by patient.

    Fractions apply approximately to groups, not rows. No stratification is
    claimed. Missing classes in training fail explicitly. Never reroll seeds
    based on validation/test scores. Existing official splits take precedence.
    """
    frame = frame.copy().reset_index(drop=True)
    if 'patient_id' not in frame or frame['patient_id'].isna().any():
        raise ValueError('Verified patient_id is required; do not substitute image IDs.')
    groups = frame['patient_id'].astype(str).str.strip()
    if groups.eq('').any():
        raise ValueError('Empty patient_id.')
    frame['patient_id'] = groups
    if 'split' not in frame:
        if groups.nunique() < 5:
            raise ValueError('At least five patients required for train/validation/test.')
        dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(frame, groups=groups))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(frame.iloc[dev], groups=groups.iloc[dev]))
        frame['split'] = 'test'
        frame.loc[dev[train], 'split'] = 'train'
        frame.loc[dev[val], 'split'] = 'validation'
    if frame['split'].isna().any() or not set(frame['split']).issubset({'train', 'validation', 'test'}):
        raise ValueError('split must be train, validation or test.')
    for key in ('patient_id', 'filename', 'pixel_sha256'):
        if key in frame and (frame.groupby(key)['split'].nunique() > 1).any():
            raise ValueError(f'{key} overlaps partitions.')
    parts = {s: frame[frame['split'].eq(s)].copy() for s in ('train', 'validation', 'test')}
    if any(p.empty for p in parts.values()):
        raise ValueError('Every partition must contain observations.')
    if 'label' in frame:
        missing = set(frame['label']) - set(parts['train']['label'])
        if missing:
            raise ValueError(f'Classes missing from training: {sorted(missing)}. Review group coverage.')
    return parts


def load_manifest(path: Path, root: Path, seed: int = 12345):
    frame = pd.read_csv(path, dtype=str)
    required = {'filename', 'label', 'patient_id'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError(f'Manifest requires nonempty columns {sorted(required)}.')
    if frame['label'].str.strip().eq('').any():
        raise ValueError('Empty label.')
    return patient_split(validate_images(frame, root), seed)


def keras_generators(parts, classes, batch_size=16, vgg=False, augment=None, seed=12345):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import preprocess_input
    kwargs = {'preprocessing_function': preprocess_input} if vgg else {'rescale': 1 / 255.}
    result = {}
    for split, frame in parts.items():
        generator = ImageDataGenerator(**kwargs, **(augment or {} if split == 'train' else {}))
        result[split] = generator.flow_from_dataframe(
            frame, x_col='filename', y_col='label', classes=classes,
            target_size=(224, 224), color_mode='rgb', class_mode='categorical',
            batch_size=batch_size, shuffle=split == 'train', seed=seed)
        if result[split].n != len(frame):
            raise ValueError('Generator silently discarded manifest rows.')
    return result


def evaluate_classifier(model, generator):
    """All rows in deterministic order; fixed class vocabulary even if absent."""
    from sklearn.metrics import classification_report, confusion_matrix
    if generator.shuffle:
        raise ValueError('Evaluation generator must use shuffle=False.')
    generator.reset()
    probabilities = model.predict(generator, verbose=0)
    if len(probabilities) != generator.n or not np.isfinite(probabilities).all():
        raise ValueError('Invalid prediction count or non-finite probabilities.')
    labels = list(range(len(generator.class_indices)))
    names = sorted(generator.class_indices, key=generator.class_indices.get)
    predictions = probabilities.argmax(axis=1)
    return {
        'classification_report': classification_report(generator.classes, predictions, labels=labels, target_names=names, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(generator.classes, predictions, labels=labels).tolist(),
        'class_names': names,
    }
