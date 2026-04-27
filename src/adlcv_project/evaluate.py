import torch
from torch.utils.data import DataLoader

from adlcv_project.heatmap_dataset import HeatmapDataset
from adlcv_project.models.resnet import MultiScaleBackbone
from adlcv_project.models.transformer import SimpleTransformer
from adlcv_project.models.model import MainModel, Decoder
from adlcv_project.visualize import show_gt_vs_pred, visualize_5_samples, plot_loss_curve


import torch
import torch.nn.functional as F
from tqdm import tqdm


def heatmap_eval_metrics(pred_logits, target_probs, topk=(1, 5, 10, 50)):

    B = pred_logits.size(0)

    pred_log_probs = F.log_softmax(pred_logits.reshape(B, -1), dim=1)
    pred_probs = pred_log_probs.exp()

    target = target_probs.reshape(B, -1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)

    kl = F.kl_div(
        pred_log_probs,
        target,
        reduction="batchmean"
    )

    metrics = {
        "kl": kl.item()
    }

    for k in topk:
        k = min(k, pred_probs.size(1))

        _, top_idx = torch.topk(pred_probs, k=k, dim=1)

        covered_mass = torch.gather(target, dim=1, index=top_idx).sum(dim=1)

        metrics[f"top{k}_coverage"] = covered_mass.mean().item()

    top1_idx = pred_probs.argmax(dim=1, keepdim=True)
    top1_gt_mass = torch.gather(target, dim=1, index=top1_idx)

    metrics["top1_gt_mass"] = top1_gt_mass.mean().item()

    return metrics

def evaluate_distribution_match(model, loader, device):
    model.eval()

    total = {
        "kl": 0.0,
        "top1_coverage": 0.0,
        "top5_coverage": 0.0,
        "top10_coverage": 0.0,
        "top50_coverage": 0.0,
        "top1_gt_mass": 0.0,
    }

    n_batches = 0

    with torch.no_grad():
        for images, class_embeds, targets, *_ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            class_embeds = class_embeds.to(device)
            targets = targets.to(device)

            pred_logits = model(images, class_embeds)

            batch_metrics = heatmap_eval_metrics(
                pred_logits=pred_logits,
                target_probs=targets,
                topk=(1, 5, 10, 50)
            )

            for key, value in batch_metrics.items():
                total[key] += value

            n_batches += 1

    avg = {key: value / n_batches for key, value in total.items()}

    print("\nEvaluation metrics:")
    for key, value in avg.items():
        print(f"{key}: {value:.6f}")

    return avg


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HeatmapDataset(
    index_path="data/preprocessed_targets_top20/train/index.json",
    data_root="data",
    class_embedding_path="data/preprocessed_targets_top20/class_embeddings.pt"
)

loader = DataLoader(dataset, batch_size=5, shuffle=True)

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

ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    images, class_embeds, targets, fg_classes, bg_paths = next(iter(loader))

    images = images.to(device)
    class_embeds = class_embeds.to(device)

    pred_logits = model(images, class_embeds)


plot_loss_curve("checkpoints/best_model.pt")

show_gt_vs_pred(
    image=images[0].cpu(),
    target=targets[0].cpu(),
    pred_logits=pred_logits[0].cpu(),
    fg_class=fg_classes[0],
)

visualize_5_samples(model, loader, device)




