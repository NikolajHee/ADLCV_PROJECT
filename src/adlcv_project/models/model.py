




from torch import nn
import torch

from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def forward(self, text_inputs):
        # text_inputs: List of strings
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = self.model(**tokenized)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation

# global injection
class FiLM(nn.Module):
    def __init__(self, class_dim, feature_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(feature_channels, affine=False)
        # Predicts both scale and shift from the class embedding
        self.mlp = nn.Linear(class_dim, feature_channels * 2)

    def forward(self, features, class_embeds):
        # features: [B, C, H, W]
        # class_embeds: [B, class_dim]
        
        # Get gamma and beta: [B, C*2]
        gamma_beta = self.mlp(class_embeds)
        
        # Split into scale and shift: [B, C, 1, 1] each
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # Apply normalization and modulation
        normalized = self.norm(features)
        return (normalized * (1 + gamma)) + beta



class MainModel(nn.Module):
    def __init__(self, backbone, transformer, decoder):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.decoder = decoder
        self.FiLM_layer = FiLM(class_dim=512, feature_channels=1024)

    def forward(self, x, class_embeds):
        features = self.backbone(x)["c4"]

        class_embeds = class_embeds.to(x.device)
        features = self.FiLM_layer(features, class_embeds)

        B, C, H, W = features.shape

        tokens = features.flatten(2).permute(0, 2, 1)
        tokens = self.transformer(tokens)

        features = tokens.permute(0, 2, 1).view(B, C, H, W)

        output = self.decoder(features)

        return output

def center_crop_512(img, img_size):
    w, h = img.size
    left, top = (w - img_size) // 2, (h - img_size) // 2
    return img.crop((left, top, left + img_size, top + img_size))

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.cnn(x)


if __name__ == "__main__":
    # Example usage
    import os 
    import numpy as np
    from PIL import Image


    DEMO_DIR = "data/"

    IMG_SIZE = 512


    example_case = {
        'entry_id': 2,
        'bg_path': 'data_large_standard/w/wave/00002144.jpg',
        'fg_class': 'surfboard',
        'bbox': [0.316406, 0.328125, 0.152344, 0.269531],
        'label': 1,
        'image_reward_score': -0.793100893497467,
        'confidence': 0.3901195526123047,
        'source': 'ho'
    }


    from adlcv_project.models.resnet import MultiScaleBackbone
    from adlcv_project.models.transformer import SimpleTransformer

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
    )

    img_og = Image.open(os.path.join(DEMO_DIR, "data_large_standard/w/wave/00002144.jpg",)).convert("RGB")
    img = center_crop_512(img_og, IMG_SIZE)
    example_input = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    example_input = example_input.unsqueeze(0)  # Add batch dimension
    
    print(f"Example input shape: {example_input.shape}")

    example_labels = example_case["fg_class"]  # Example class label
    output = model(example_input, example_labels)

    print(output.shape)