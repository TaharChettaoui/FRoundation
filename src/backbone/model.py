import torch
import logging
import clip

class ClipModel():
    def __init__(self, rank, model_name):
        self.backbone, _ = clip.load(model_name, device="cuda", jit=False)
        self.backbone.to(rank)

        for param in self.backbone.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)
