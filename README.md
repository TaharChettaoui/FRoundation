# FRoundation: Are Foundation Models Ready for Face Recognition? 

Tahar Chettaoui,
Naser Damer,
Fadi Boutros

[[`Paper`](https://arxiv.org/abs/2410.23831)] [[`Blog`](https://taharchettaoui.github.io/FRoundation_web/)]

PyTorch implementation and pretrained models for **FRoundation**.

### Pretrained models

| Method    | Backbone | Train data | Link |
| -------- | ------- | ------- | ------- |
| Baseline  | <br>Small <br><br><br><br> Large <br><br>  | CASIA-WebFace <br> MS1MV2 <br> WebFace4M <br><br> CASIA-WebFace <br> MS1MV2 <br> WebFace4M| [Download](https://owncloud.fraunhofer.de/index.php/s/RqDWPal8qvXKuH7) <br> [Download](https://owncloud.fraunhofer.de/index.php/s/7HubejKcXwhxk8D) <br> [Download](https://owncloud.fraunhofer.de/index.php/s/ZXnzAODROxWncil) <br><br> ... <br> ... <br> ...|
| CLIP  | <br>Base <br><br><br><br> Large <br><br>  | CASIA-WebFace <br> MS1MV2 <br> WebFace4M <br><br> CASIA-WebFace <br> MS1MV2 <br> WebFace4M|[Download](https://owncloud.fraunhofer.de/index.php/s/oeyZTsXKYKFID5M) <br> [Download](https://owncloud.fraunhofer.de/index.php/s/OyD0N0KYvyToBBr) <br> [Download](https://owncloud.fraunhofer.de/index.php/s/p2ZbWOsp1nVLv0f) <br><br> ... <br> ... <br> ...|
| DINOv2  | <br>Small <br><br><br><br> Large <br><br>  | CASIA-WebFace <br> MS1MV2 <br> WebFace4M <br><br> CASIA-WebFace <br> MS1MV2 <br> WebFace4M|... <br> ... <br> ... <br><br> ... <br> ... <br> ...|

### Usage
First, install all the dependencies listed in the requirements.txt file. This code has been tested with Python 3.9. Below is an example of how to load the model and pretrained weights, specifically for CLIP ViT-B/16.

```
import sys
import os
import torch
import torch.distributed as dist

from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel

from backbone import get_model
from finetuning import apply_lora_model
from data.transform import transform_image
from utils.evaluation import CallBackVerification


# Load pretrained model
model = get_model(rank, model_name="clip", backbone_size='ViT-B/16')

# Attach LoRA layers
apply_lora_model(
    rank, 
    model, 
    training_type="image_encoder_only",
    model_name="clip",
    backbone_size='ViT-B/16', 
    lora_target_modules=['q', 'v'],
    lora_r=16, 
    lora_a=32, 
    lora_dropout=0.25, 
    device=rank, 
    position="all"
)

# Load pretrained model
model_path = "path_to_pretrained_model" 
model.backbone.load_state_dict(torch.load(model_path))
model.backbone.visual # To access the image encoder of CLIP
```

### Training
After updating the config.config.py file with the required paths and desired parameters, you can initiate training using:

```
./train.sh
```

In addition to logs that document the loss and hyperparameters during training, a **TensorBoard** with various plots is also available.

### Coming soon ...
- [ ] Pretrained models link
- [ ] Dinov2Training pipeline
- [ ] Evaluation code

### Citation

```
@misc{chettaoui2024froundationfoundationmodelsready,
      title={FRoundation: Are Foundation Models Ready for Face Recognition?}, 
      author={Tahar Chettaoui and Naser Damer and Fadi Boutros},
      year={2024},
      eprint={2410.23831},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.23831}, 
}
```

### License 

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```

