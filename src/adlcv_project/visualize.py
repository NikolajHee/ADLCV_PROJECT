import matplotlib.pyplot as plt
import numpy as np
import torch


def show_gt_vs_pred(image, target, pred_logits, fg_class, alpha=0.45, cmap="jet"):
    """
    image: [3, H, W] torch tensor
    target: [S, G, G] torch tensor
    pred_logits: [S, G, G] torch tensor
    """

    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)

    target_2d = target.sum(dim=0).cpu().numpy()

    pred_probs = torch.softmax(pred_logits.flatten(), dim=0).view_as(pred_logits)
    pred_2d = pred_probs.sum(dim=0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_np)
    axes[0].imshow(
        target_2d,
        cmap=cmap,
        alpha=alpha,
        extent=(0, image_np.shape[1], image_np.shape[0], 0),
        interpolation="bilinear",
    )
    axes[0].set_title(f"Ground truth\nclass={fg_class}")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(
        pred_2d,
        cmap=cmap,
        alpha=alpha,
        extent=(0, image_np.shape[1], image_np.shape[0], 0),
        interpolation="bilinear",
    )
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



def visualize_5_samples(model, loader, device, alpha=0.45, cmap="jet"):
    model.eval()

    batch = next(iter(loader))
    images, class_embeds, targets, fg_classes, bg_paths = batch
    print("Batch classes:", fg_classes)

    images = images.to(device)
    class_embeds = class_embeds.to(device)

    with torch.no_grad():
        pred_logits = model(images, class_embeds)

        pred_probs = torch.softmax(
            pred_logits.reshape(pred_logits.size(0), -1),
            dim=1
        ).reshape_as(pred_logits)

    n = min(5, images.size(0))

    fig, axes = plt.subplots(
        2, n,
        figsize=(3.2 * n, 6.5),
        constrained_layout=True
    )

    for i in range(n):
        print(i, fg_classes[i], bg_paths[i])
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)

        gt_map = targets[i].sum(dim=0).detach().cpu().numpy()
        pred_map = pred_probs[i].sum(dim=0).detach().cpu().numpy()

        axes[0, i].imshow(image)
        axes[0, i].imshow(
            gt_map,
            cmap=cmap,
            alpha=alpha,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        axes[0, i].set_title(
            f"GT: {fg_classes[i]}\n{bg_paths[i].split('/')[-1]}",
            fontsize=9
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(image)
        axes[1, i].imshow(
            pred_map,
            cmap=cmap,
            alpha=alpha,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        axes[1, i].set_title("Prediction", fontsize=10)
        axes[1, i].axis("off")

    plt.show()

import matplotlib.pyplot as plt


def plot_loss_curve(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    losses = ckpt["epoch_losses"]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.show()