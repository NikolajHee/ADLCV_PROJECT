import torch
from torch.utils.data import DataLoader

from adlcv_project.heatmap_dataset import HeatmapDataset
from adlcv_project.models.resnet import MultiScaleBackbone
from adlcv_project.models.transformer import SimpleTransformer
from adlcv_project.models.model import MainModel, Decoder
from adlcv_project.visualize import show_gt_vs_pred, visualize_5_samples


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HeatmapDataset(
    index_path="data/preprocessed_targets/train/index.json",
    data_root="data",
    class_embedding_path="data/preprocessed_targets/class_embeddings.pt"
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
    images, class_embeds, targets, fg_classes = next(iter(loader))

    images = images.to(device)
    class_embeds = class_embeds.to(device)

    pred_logits = model(images, class_embeds)

show_gt_vs_pred(
    image=images[0].cpu(),
    target=targets[0].cpu(),
    pred_logits=pred_logits[0].cpu(),
    fg_class=fg_classes[0],
)

visualize_5_samples(model, loader, device)
