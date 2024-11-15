from easydict import EasyDict as edict
import torch.distributed as dist

config = edict()

# Training
config.num_epoch = 40
config.global_batch_size = 512
config.lr_model = 0.0001
config.lr_header = 0.0001
config.momentum = 0.9
config.weight_decay = 0.05
config.max_norm = 5
config.loss = "CosFace" # ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus, AdaFace
config.global_step = 0

# Scheduler
config.scheduler_type = "cosine" # lambda, cosine
config.warmup = True
config.num_warmup_epochs = 5
config.T_0 = 5
config.T_mult = 2
config.eta_min = 1e-6
config.lr_func_drop = [22, 30, 40]

# Model
config.model_name = "clip"
config.training_desc = "Training_description"
if config.model_name == "clip":
    config.backbone_size = "ViT-B/16"  #"ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    config.training_type = "image_encoder_only" # "text_image_header", "text_image_contrastive", "image_encoder_only"

# LoRA
config.use_lora = False
config.lora_r = 16 # 2, 4, 8, 16
config.lora_a = 16 # 2 - 512
config.lora_dropout = 0.25
config.lora_bias = "none" # "none", "all", "lora_only"
config.rslora = True
if config.model_name == "clip":
    config.lora_target_modules = ['q', 'v'] # ['q', 'k', 'v', 'o']

# Logging
config.output_path = "output/training/" + config.model_name + "_" + config.training_desc
config.log_every = 50

# Preprocessing and data augmentation
config.horizontal_flip = True
config.rand_augment = True
config.image_size = 224
if config.model_name == "dinov2":
    config.normalize_type = "imagenet"
    config.interpolation_type = "bicubic"
elif config.model_name == "clip":
    config.normalize_type = "clip"
    config.interpolation_type = "bicubic"

# Dataset (Training)
# Real:         "MS1MV2" / "casia_webface" / "WEBFACE4M"
# Synthetic:    "Idifface" /  " "
config.dataset_name = "casia_webface"
if config.dataset_name == "casia_webface":
    config.dataset_path = "/home/chettaou/data/casia_training"
    config.selective_dataset = False
    config.num_classes = 10572
    config.num_image = 501195
elif config.dataset_name == "MS1MV2":
    config.dataset_path = "Training_data_path"
    config.num_classes = 85742
    config.num_image = 5822653
elif config.dataset_name == "WEBFACE4M":
    config.dataset_path = "Training_data_path"
    config.num_classes = 205990
    config.num_image = 4235242
elif config.dataset_name == "Idifface":
    config.dataset_path = "Training_data_path"
    config.num_classes = 10000 # 10049
    config.num_image = 502403 

# Evaluation
# "lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"
# RFW: "African_test", "Asian_test", "Caucasian_test", "Indian_test"
config.eval_path = "/home/chettaou/data/validation"
config.val_targets = ["lfw"] # , "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.eval_every = 5
config.save_every = 10
config.batch_size_eval = 16

# Optimizer
if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif (config.loss=="AdaFace"):
    config.s = 64.0
    config.m = 0.4
    config.h = 0.333
    config.std = 0.05
else:
    config.s = 64.0
    config.m = 0.3
    config.std = 0.05
