import logging

from .lora import apply_lora_clip
from .loralib.utils import mark_only_lora_as_trainable
from utils.utils import print_trainable_parameters

def apply_lora_model(rank, model, **kwargs):
    name = kwargs["model_name"]
    
    logging.info("Before LoRA : " + print_trainable_parameters(model.backbone))

    if name == "clip":
        logging.info("Add LoRA layers ...")
        apply_lora_clip(
            model=model, 
            training_type=kwargs["training_type"], 
            model_name=kwargs["backbone_size"], 
            target_modules=kwargs["lora_target_modules"],
            lora_rank=kwargs["lora_r"], 
            lora_alpha=kwargs["lora_a"], 
            lora_dropout=kwargs["lora_dropout"], 
            device=rank, 
            position="all"
        )
        mark_only_lora_as_trainable(model.backbone)
    else:
        raise ValueError()
    
    logging.info("After LoRA : " + print_trainable_parameters(model.backbone))