"""Train the binary U-Net on paired 8-bit image and mask tiles."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from unet import UNet

IMAGE_SIZE = (256, 256)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class PairedTileDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        images = {path.stem: path for path in images_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS}
        masks = {path.stem: path for path in masks_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS}
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks or missing_images:
            raise ValueError(f"Unpaired files; missing masks={missing_masks}, missing images={missing_images}")
        self.pairs = [(images[key], masks[key]) for key in sorted(images)]
        if not self.pairs:
            raise ValueError("No paired image/mask tiles found")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        return (
            torch.from_numpy(image_array).unsqueeze(0),
            torch.from_numpy(mask_array).unsqueeze(0),
        )


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def evaluate(model: UNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    true_positive = false_positive = false_negative = true_negative = 0
    with torch.inference_mode():
        for images, masks in loader:
            predictions = torch.sigmoid(model(images.to(device))) >= 0.5
            targets = masks.to(device) >= 0.5
            true_positive += int((predictions & targets).sum().item())
            false_positive += int((predictions & ~targets).sum().item())
            false_negative += int((~predictions & targets).sum().item())
            true_negative += int((~predictions & ~targets).sum().item())
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "iou": true_positive / union if union else 1.0,
        "dice": 2 * true_positive / dice_denominator if dice_denominator else 1.0,
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "pixel_accuracy": (true_positive + true_negative) / max(1, true_positive + false_positive + false_negative + true_negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of preprocessed image tiles")
    parser.add_argument("--masks", type=Path, required=True, help="Directory of binary masks with matching file stems")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-masks", type=Path)
    parser.add_argument("--test-images", type=Path)
    parser.add_argument("--test-masks", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "model" / "oiltrace_unet.pth")
    args = parser.parse_args()

    dataset = PairedTileDataset(args.images, args.masks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if bool(args.val_images) != bool(args.val_masks) or bool(args.test_images) != bool(args.test_masks):
        parser.error("Provide both image and mask directories for each validation/test split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    validation_loader = None
    test_loader = None
    if args.val_images:
        validation_loader = DataLoader(PairedTileDataset(args.val_images, args.val_masks), batch_size=args.batch_size)
    if args.test_images:
        test_loader = DataLoader(PairedTileDataset(args.test_images, args.test_masks), batch_size=args.batch_size)
    best_validation_iou = -1.0
    best_epoch = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        message = f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(dataset):.4f}"
        validation_metrics = evaluate(model, validation_loader, device) if validation_loader else None
        if validation_metrics:
            message += f" val_iou={validation_metrics['iou']:.4f} val_pixel_accuracy={validation_metrics['pixel_accuracy']:.4f}"
        print(message)
        current_iou = validation_metrics["iou"] if validation_metrics else float(epoch + 1)
        if current_iou > best_validation_iou:
            best_validation_iou = current_iou
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.cpu().state_dict(),
                    "in_channels": 1,
                    "base_channels": args.base_channels,
                    "epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                },
                args.output,
            )
            model.to(device)

    if test_loader:
        checkpoint = torch.load(args.output, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"test_metrics={json.dumps(evaluate(model, test_loader, device), sort_keys=True)}")
    print(f"best_epoch={best_epoch} best_validation_iou={best_validation_iou:.4f}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()