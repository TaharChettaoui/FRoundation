import torch

from peft import LoraConfig, LoraModel
from .loralib.layers import PlainMultiheadAttentionLoRA


def apply_lora_clip(model, training_type, model_name, target_modules, lora_rank, lora_alpha, lora_dropout, device, position="all"):
    list_lora_layers = []

    assert training_type in ["text_image_header", "text_image_contrastive", "image_encoder_only"]

    if training_type in ["text_image_header", "text_image_contrastive", "image_encoder_only"]:
        vision_encoder = model.backbone.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            for name, submodule in block.named_children():
                if isinstance(submodule, torch.nn.MultiheadAttention):
                    pass
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=target_modules, r=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout).to(device)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)

    if training_type in ["text_image_header", "text_image_contrastive"]:
        text_encoder = model.backbone.transformer
        for i, block in enumerate(text_encoder.resblocks):
            for name, submodule in block.named_children():
                if isinstance(submodule, torch.nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=target_modules, r=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout).to(device)
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)
    
    return list_lora_layers


def apply_lora_peft(rank, model, lora_r, lora_a, lora_target_modules, lora_dropout, use_rslora):

    lora_config = LoraConfig(
        task_type="FEATURE_EXTRACTION", # "TOKEN_CLS"
        r=lora_r,
        lora_alpha=lora_a,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
    )

    return LoraModel(model.backbone, lora_config, "default").to(rank)
