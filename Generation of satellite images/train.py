"""Paired summer-to-winter satellite image translation.

Example:
  python train.py --data-dir /path/to/images --epochs 20 --output-dir results

Expected files: <city>_summer.jpg and <city>_winter.jpg.
Cities are split before tiles are generated to prevent location leakage.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp


@dataclass(frozen=True)
class Tile:
    city: str
    summer: Path
    winter: Path
    box: tuple[int, int, int, int]


def pair_cities(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Find matching JPEG season pairs; report incomplete pairs explicitly."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    pairs = {}
    summer_files = sorted(data_dir.glob("*_summer.jpg"))
    winter_files = sorted(data_dir.glob("*_winter.jpg"))
    summers = {p.name.removesuffix("_summer.jpg"): p for p in summer_files}
    winters = {p.name.removesuffix("_winter.jpg"): p for p in winter_files}
    missing = sorted(set(summers) ^ set(winters))
    if missing:
        raise ValueError(f"Missing seasonal counterpart for cities: {missing}")
    for city in sorted(set(summers) & set(winters)):
        pairs[city] = (summers[city], winters[city])
    if len(pairs) < 2:
        raise ValueError("At least two paired cities are needed for a city-level split.")
    return pairs


def split_cities(
    pairs: dict[str, tuple[Path, Path]], val_fraction: float, seed: int
) -> tuple[dict[str, tuple[Path, Path]], dict[str, tuple[Path, Path]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    cities = sorted(pairs)
    random.Random(seed).shuffle(cities)
    count = max(1, min(len(cities) - 1, round(len(cities) * val_fraction)))
    val = set(cities[:count])
    return ({c: pairs[c] for c in cities if c not in val},
            {c: pairs[c] for c in cities if c in val})


def build_tiles(pairs: dict[str, tuple[Path, Path]], rows: int, cols: int) -> list[Tile]:
    if rows < 1 or cols < 1:
        raise ValueError("Rows and columns must be positive.")
    tiles: list[Tile] = []
    for city, (summer, winter) in pairs.items():
        with Image.open(summer) as s, Image.open(winter) as w:
            if s.size != w.size:
                raise ValueError(f"Unaligned image sizes for {city}: {s.size} vs {w.size}")
            width, height = s.size
        if width < cols or height < rows:
            raise ValueError(f"Image of {city} is smaller than tile grid.")
        # Pixel boundaries partition the entire picture, retaining edge pixels.
        xs = [width * j // cols for j in range(cols + 1)]
        ys = [height * i // rows for i in range(rows + 1)]
        for i in range(rows):
            for j in range(cols):
                tiles.append(Tile(city, summer, winter,
                                  (xs[j], ys[i], xs[j + 1], ys[i + 1])))
    return tiles


class SeasonalTiles(Dataset):
    def __init__(self, tiles: list[Tile], image_size: int = 128):
        self.tiles = tiles
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile = self.tiles[index]
        # Open within each worker: no shared file handle across DataLoader workers.
        with Image.open(tile.summer) as src, Image.open(tile.winter) as target:
            src_crop = src.convert("RGB").crop(tile.box)
            target_crop = target.convert("RGB").crop(tile.box)
            src_crop = src_crop.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            target_crop = target_crop.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return TF.to_tensor(src_crop), TF.to_tensor(target_crop)


class SeasonalTranslator(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=3, activation=None,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.unet(images).sigmoid()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    sum_abs = 0.0
    sum_sq = 0.0
    pixel_count = 0
    for summer, winter in loader:
        prediction = model(summer.to(device))
        delta = prediction - winter.to(device)
        sum_abs += delta.abs().sum().item()
        sum_sq += delta.square().sum().item()
        pixel_count += delta.numel()
    if not pixel_count:
        raise ValueError("Empty validation loader.")
    return sum_abs / pixel_count, sum_sq / pixel_count


@torch.no_grad()
def save_example(model: nn.Module, dataset: SeasonalTiles, device: torch.device, output: Path):
    model.eval()
    summer, winter = dataset[0]
    prediction = model(summer.unsqueeze(0).to(device)).cpu().squeeze(0)
    # Three frames: input summer / true winter / generated winter.
    panel = torch.cat((summer, winter, prediction), dim=2).clamp(0, 1)
    TF.to_pil_image(panel).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--pretrained", action="store_true", help="Download ImageNet encoder weights")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs and batch-size must be positive, image-size must be >= 32")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pairs = pair_cities(args.data_dir)
    train_pairs, val_pairs = split_cities(pairs, args.val_fraction, args.seed)
    train_data = SeasonalTiles(build_tiles(train_pairs, args.rows, args.cols), args.image_size)
    val_data = SeasonalTiles(build_tiles(val_pairs, args.rows, args.cols), args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeasonalTranslator(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    history: list[dict] = []
    best_mae = float("inf")

    print(f"Training cities: {sorted(train_pairs)}; validation cities: {sorted(val_pairs)}")
    print(f"Training tiles: {len(train_data)}; validation tiles: {len(val_data)}; device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for summer, winter in train_loader:
            summer, winter = summer.to(device), winter.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(summer)
            loss = criterion(prediction, winter)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * summer.size(0)
            samples += summer.size(0)
        val_mae, val_mse = evaluate(model, val_loader, device)
        metrics = {"epoch": epoch, "train_mae": loss_sum / samples,
                   "val_mae": val_mae, "val_mse": val_mse}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                        "model": "Unet/resnet34", "config": vars(args)},
                       args.output_dir / "best_model.pt")
            save_example(model, val_data, device, args.output_dir / "validation_example.png")

    report = {"train_cities": sorted(train_pairs), "validation_cities": sorted(val_pairs),
              "train_tiles": len(train_data), "validation_tiles": len(val_data),
              "history": history, "best_val_mae": best_mae,
              "note": "MAE/MSE computed on paired winter targets from held-out cities; not a perceptual metric."}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
