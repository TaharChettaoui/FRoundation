import torch
import logging
import clip

from transformers import Dinov2Model, Dinov2Config

class ClipModel():
    def __init__(self, rank, backbone_size):
        self.backbone, _ = clip.load(backbone_size, device="cuda", jit=False)
        self.backbone.to(rank)

        for param in self.backbone.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

class DINOv2Model():
    def __init__(self, rank, backbone_size):
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-" + backbone_size)
        self.backbone.to(rank)


class Dinov2BaselineModel():
    def __init__(self, rank, backbone_size):
        logging.info("Loading scratch vit " + backbone_size + " ...")

        if backbone_size == "small":
            configuration = Dinov2Config(
                hidden_size = 384,
                num_hidden_layers = 12,
                num_attention_heads = 6,
                mlp_ratio = 4,
                hidden_act = 'gelu',
                layer_norm_eps = 1e-06,
                image_size = 518,
                patch_size = 14,
                num_channels = 3,
                qkv_bias = True,
                layerscale_value = 1.0,
                torch_dtype = torch.float32,
                transformers_version = "4.31.0.dev0",
            )

        elif backbone_size == "base":
            configuration = Dinov2Config(
                hidden_size = 768,
                num_hidden_layers = 12,
                num_attention_heads = 12,
                mlp_ratio = 4,
                hidden_act = 'gelu',
                layer_norm_eps = 1e-06,
                image_size = 518,
                patch_size = 14,
                num_channels = 3,
                qkv_bias = True,
                layerscale_value = 1.0,
                torch_dtype = torch.float32,
                transformers_version = "4.31.0.dev0",
            )

        elif backbone_size == "large":
            configuration = Dinov2Config(
                hidden_size = 1024,
                num_hidden_layers = 24,
                num_attention_heads = 16,
                mlp_ratio = 4,
                hidden_act = 'gelu',
                layer_norm_eps = 1e-06,
                image_size = 518,
                patch_size = 14,
                num_channels = 3,
                qkv_bias = True,
                layerscale_value = 1.0,
                torch_dtype = torch.float32,
                transformers_version = "4.31.0.dev0",
            )
        elif backbone_size == "giant":
            configuration = Dinov2Config(
                hidden_size = 1536,
                num_hidden_layers = 40,
                num_attention_heads = 24,
                mlp_ratio = 4,
                hidden_act = 'gelu',
                layer_norm_eps = 1e-06,
                image_size = 518,
                patch_size = 14,
                num_channels = 3,
                qkv_bias = True,
                layerscale_value = 1.0,
                use_swiglu_ffn = True,
                torch_dtype = torch.float32,
                transformers_version = "4.31.0.dev0",
            )

        # Initializing a model (with random weights)
        self.backbone = Dinov2Model(configuration)
        self.backbone.to(rank)
