import logging
from .model import ClipModel, DINOv2Model, Dinov2BaselineModel

def get_model(rank, **kwargs):
    name = kwargs["model_name"]

    if name == "clip":
        logging.info("Loading model: " + name + " " + kwargs["backbone_size"])

        clip_model = ClipModel(
            rank=rank, 
            backbone_size=kwargs["backbone_size"]
        )
        return clip_model
    
    elif name == "dinov2":
        dinov2_model = DINOv2Model(
            rank=rank,
            backbone_size=kwargs["backbone_size"], 
        )
        return dinov2_model
    
    elif name == "baseline":
        dinov2_baseline_model = Dinov2BaselineModel(
            rank=rank,
            backbone_size=kwargs["backbone_size"],
        )
        return dinov2_baseline_model
    
    else:
        raise ValueError()


def get_output_dim(**kwargs):
    name = kwargs["model_name"]

    if name == "clip":
        backbone_embeddings = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
        }
        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))
        return backbone_embeddings[kwargs["backbone_size"]]
    
    elif name == "dinov2" or name == "baseline":
        backbone_embeddings = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))
        return backbone_embeddings[kwargs["backbone_size"]]
    
    else:
        raise ValueError()
