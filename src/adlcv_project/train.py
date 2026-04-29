import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from adlcv_project.heatmap_dataset import HeatmapDataset
from adlcv_project.models.resnet import MultiScaleBackbone
from adlcv_project.models.transformer import SimpleTransformer
from adlcv_project.models.model import MainModel, Decoder, TextEncoder


def build_class_embedding_cache(index_path, save_path, device):
    if os.path.exists(save_path):
        print(f"Using cached class embeddings: {save_path}")
        return

    print("Building class embedding cache...")

    with open(index_path, "r") as f:
        index = json.load(f)

    classes = sorted({item["fg_class"] for item in index})

    text_encoder = TextEncoder().to(device)
    text_encoder.eval()

    class_embeddings = {}

    with torch.no_grad():
        for cls in classes:
            emb = text_encoder([cls]).squeeze(0).cpu()
            class_embeddings[cls] = emb

    torch.save(class_embeddings, save_path)
    print(f"Saved class embeddings to: {save_path}")


def compute_kl_loss(pred_logits, targets):
    pred_log_probs = F.log_softmax(
        pred_logits.flatten(1),
        dim=1,
    )

    target_probs = targets.flatten(1)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    loss = F.kl_div(
        pred_log_probs,
        target_probs,
        reduction="batchmean",
    )

    return loss


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for images, class_embeds, targets, fg_classes, bg_paths in tqdm(val_loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        class_embeds = class_embeds.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        pred_logits = model(images, class_embeds)
        loss = compute_kl_loss(pred_logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)

def train():
    num_epochs = 20
    batch_size = 16

    
    patience = 25
    min_delta = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    train_index_path = "data/preprocessed_targets/train/index.json"
    val_index_path = "data/preprocessed_targets/val/index.json"
    class_embedding_path = "data/preprocessed_targets/class_embeddings.pt"

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")

    build_class_embedding_cache(
        index_path=train_index_path,
        save_path=class_embedding_path,
        device=device,
    )

    train_dataset = HeatmapDataset(
        index_path=train_index_path,
        data_root="data",
        class_embedding_path=class_embedding_path,
        augment=True,
    )

    val_dataset = HeatmapDataset(
        index_path=val_index_path,
        data_root="data",
        class_embedding_path=class_embedding_path,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    backbone = MultiScaleBackbone()

    transformer = SimpleTransformer(
        embed_dim=1024,
        num_heads=8,
        num_layers=2,
        max_seq_len=32 * 32,
        pool=None,
        dropout=0.2,
    )

    decoder = Decoder(input_dim=1024, output_dim=8)

    model = MainModel(
        backbone=backbone,
        transformer=transformer,
        decoder=decoder,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=3e-4,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = -1

    train_losses = []
    val_losses = []
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, class_embeds, targets, fg_classes, bg_paths in pbar:
            images = images.to(device, non_blocking=True)
            class_embeds = class_embeds.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_logits = model(images, class_embeds)
                loss = compute_kl_loss(pred_logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            avg_train_loss = running_loss / num_batches

            pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                train_loss=f"{avg_train_loss:.4f}",
                best_val=f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "inf",
            )

        train_loss = running_loss / max(num_batches, 1)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss - min_delta

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        latest_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "epochs_without_improvement": epochs_without_improvement,
        }

        torch.save(latest_checkpoint, latest_path)
        print(f"Saved latest checkpoint to {latest_path}")

        if improved:
            torch.save(latest_checkpoint, best_path)
            print(f"Saved new best model to {best_path} | val_loss={best_val_loss:.4f}")
        else:
            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{patience} epochs"
            )

        print(
            f"Epoch {epoch + 1} done | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"best_val_loss={best_val_loss:.4f} | "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training done. Best validation loss: {best_val_loss:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Latest checkpoint saved to: {latest_path}")
    print(f"Best checkpoint saved to: {best_path}")

if __name__ == "__main__":
    train()