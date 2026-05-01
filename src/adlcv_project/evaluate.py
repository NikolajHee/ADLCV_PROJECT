import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from adlcv_project.heatmap_dataset import HeatmapDataset
from adlcv_project.models.resnet import MultiScaleBackbone
from adlcv_project.models.transformer import SimpleTransformer
from adlcv_project.models.model import MainModel, Decoder
from adlcv_project.visualize import plot_train_val_gt_pred


def plot_losses_from_checkpoint(ckpt):
    train_losses = ckpt.get("train_losses", None)
    val_losses = ckpt.get("val_losses", None)

    if train_losses is None:
        train_losses = ckpt.get("epoch_losses", None)

    if train_losses is None:
        print("No train losses found in checkpoint.")
        print("Available checkpoint keys:", ckpt.keys())
        return

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    
    plt.plot(epochs, train_losses, marker="o", label="Train loss")

    if val_losses is not None:
        plt.plot(epochs, val_losses, marker="o", label="Validation loss")

        
        val_losses_tensor = torch.tensor(val_losses)
        best_epoch = int(torch.argmin(val_losses_tensor)) + 1
        best_val = float(val_losses_tensor.min())

        
        val_color = plt.gca().lines[1].get_color()

        plt.axvline(
            best_epoch,
            linestyle="--",
            alpha=0.6,
            color=val_color,
            label=f"lowest val loss is {best_val:.2f} in epoch {best_epoch}"
        )

    
    plt.xticks(epochs)

    plt.xlabel("Epoch")
    plt.ylabel("KL loss")
    plt.legend()
    plt.grid(True, alpha=0.1)
    plt.ylim(0, max(train_losses + (val_losses if val_losses is not None else [])) * 1.1)
    plt.xlim(1, len(train_losses) + 1)
    plt.tight_layout()
    plt.show()

def build_loader(index_path, class_embedding_path, batch_size, shuffle):
    dataset = HeatmapDataset(
        index_path=index_path,
        data_root="data",
        class_embedding_path=class_embedding_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    return loader


def build_model(device):
    backbone = MultiScaleBackbone()

    transformer = SimpleTransformer(
        embed_dim=1024,
        num_heads=8,
        num_layers=2,
        max_seq_len=32 * 32,
        pool=None,
    )

    decoder = Decoder(input_dim=1024, output_dim=8)

    model = MainModel(
        backbone=backbone,
        transformer=transformer,
        decoder=decoder,
    ).to(device)

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 32

    preprocess_dir = "data/preprocessed_targets"
    checkpoint_path = "checkpoints/best_model.pt"

    train_loader = build_loader(
        index_path=f"{preprocess_dir}/train/index.json",
        class_embedding_path=f"{preprocess_dir}/class_embeddings.pt",
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = build_loader(
        index_path=f"{preprocess_dir}/val/index.json",
        class_embedding_path=f"{preprocess_dir}/class_embeddings.pt",
        batch_size=batch_size,
        shuffle=True,
    )

    model = build_model(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded checkpoint.")
    print("Checkpoint keys:", ckpt.keys())

    # 1. Plot train/validation loss curves
    plot_losses_from_checkpoint(ckpt)

    # 2. Plot qualitative train vs validation comparison
    plot_train_val_gt_pred(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_idx=0,
        val_idx=0,
        alpha=0.45,
        cmap="hot",
    )


if __name__ == "__main__":
    main()