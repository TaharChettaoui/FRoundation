import logging
from .model import ClipModel

def get_model(rank, **kwargs):
    name = kwargs["model_name"]

    if name == "clip":
        logging.info("Loading model: " + name + " " + kwargs["backbone_size"])

        clip_model = ClipModel(
            rank=rank, 
            model_name=kwargs["backbone_size"]
        )
        return clip_model
    
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
    else:
        raise ValueError()
