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


def train():
    num_epochs = 5
    batch_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    index_path = "data/preprocessed_targets/train/index.json"
    class_embedding_path = "data/preprocessed_targets/class_embeddings.pt"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    build_class_embedding_cache(
        index_path=index_path,
        save_path=class_embedding_path,
        device=device
    )

    dataset = HeatmapDataset(
        index_path=index_path,
        data_root="data",
        class_embedding_path=class_embedding_path
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    backbone = MultiScaleBackbone()

    transformer = SimpleTransformer(
        embed_dim=1024,
        num_heads=8,
        num_layers=2,
        max_seq_len=32 * 32,
        pool=None
    )

    decoder = Decoder(input_dim=1024, output_dim=8)

    model = MainModel(
        backbone=backbone,
        transformer=transformer,
        decoder=decoder
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, class_embeds, targets, fg_classes in pbar:
            images = images.to(device, non_blocking=True)
            class_embeds = class_embeds.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_logits = model(images, class_embeds)

                pred_log_probs = F.log_softmax(
                    pred_logits.flatten(1),
                    dim=1
                )

                target_probs = targets.flatten(1)
                target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

                loss = F.kl_div(
                    pred_log_probs,
                    target_probs,
                    reduction="batchmean"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            avg_loss = running_loss / num_batches

            pbar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                best_loss=f"{best_loss:.4f}" if best_loss < float("inf") else "inf"
            )

        epoch_loss = running_loss / num_batches
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} done | avg_loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": best_loss,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "epoch_losses": epoch_losses,
            }, best_path)

            print(f"Saved new best model to {best_path} | loss={best_loss:.4f}")

    print(f"Training done. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()

    from adlcv_project.visualize import plot_loss_curve

    plot_loss_curve("checkpoints/best_model.pt")

