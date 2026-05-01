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




def plot_loss_curve(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    train_losses = ckpt.get("train_losses", None)
    val_losses = ckpt.get("val_losses", None)

    if train_losses is None:
        print("No train_losses found in checkpoint.")
        return

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")

    if val_losses is not None:
        plt.plot(epochs, val_losses, label="Val Loss", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def _to_2d_maps(target, pred_logits):
    """
    target: [S, G, G]
    pred_logits: [S, G, G]
    """

    gt_map = target.sum(dim=0).detach().cpu().numpy()

    pred_probs = torch.softmax(
        pred_logits.flatten(),
        dim=0
    ).view_as(pred_logits)

    pred_map = pred_probs.sum(dim=0).detach().cpu().numpy()

    return gt_map, pred_map


def plot_train_val_gt_pred(
    model,
    train_loader,
    val_loader,
    device,
    train_idx=0,
    val_idx=0,
    alpha=0.45,
    cmap="jet",
):
    """
    Plots:
        train GT | train prediction
        val GT   | val prediction
    """

    model.eval()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    train_images, train_class_embeds, train_targets, train_fg_classes, train_bg_paths = train_batch
    val_images, val_class_embeds, val_targets, val_fg_classes, val_bg_paths = val_batch

    train_images_device = train_images.to(device)
    train_class_embeds = train_class_embeds.to(device)

    val_images_device = val_images.to(device)
    val_class_embeds = val_class_embeds.to(device)

    with torch.no_grad():
        train_logits = model(train_images_device, train_class_embeds)
        val_logits = model(val_images_device, val_class_embeds)

    train_image = train_images[train_idx].permute(1, 2, 0).cpu().numpy()
    val_image = val_images[val_idx].permute(1, 2, 0).cpu().numpy()

    train_image = np.clip(train_image, 0, 1)
    val_image = np.clip(val_image, 0, 1)

    train_gt, train_pred = _to_2d_maps(
        train_targets[train_idx],
        train_logits[train_idx].cpu()
    )

    val_gt, val_pred = _to_2d_maps(
        val_targets[val_idx],
        val_logits[val_idx].cpu()
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), tight_layout=True)

    plots = [
        (axes[0, 0], train_image, train_gt,
         f"Train GT\nclass={train_fg_classes[train_idx]}"),

        (axes[1, 0], train_image, train_pred,
         "Train prediction"),

        (axes[0, 1], val_image, val_gt,
         f"Validation GT\nclass={val_fg_classes[val_idx]}"),

        (axes[1, 1], val_image, val_pred,
         "Validation prediction"),
    ]

    for ax, image, heatmap, title in plots:
        ax.imshow(image)
        ax.imshow(
            heatmap,
            cmap=cmap,
            alpha=alpha,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Training vs validation qualitative comparison", fontsize=14)
    plt.show()