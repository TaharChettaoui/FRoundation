from easydict import EasyDict as edict

config = edict()
config.model_path = None

# Model
config.model_name = "baseline"
config.training_desc = "test_eval"
if config.model_name == "clip":
    config.backbone_size = "ViT-B/16"  # ! "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    config.training_type = "image_encoder_only" # "text_image_header", "image_encoder_only"
elif config.model_name == "dinov2":
    config.backbone_size = "small" # ! "small", "base", "large", "giant"
    config.training_type = "default"
elif config.model_name == "baseline":
    config.backbone_size = "base" # ! "small", "base", "large", "giant"
    config.training_type = "default"
# LoRA
config.use_lora = False
config.lora_r = 16 # 2, 4, 8, 16
config.lora_a = 16 # 2 - 512
config.lora_dropout = 0.25
config.lora_bias = "none"
config.rslora = True
if config.model_name == "clip":
    config.lora_target_modules = ['q', 'v']
if config.model_name == "dinov2":
    config.lora_target_modules = ['query', 'value']

# logging
config.output = "output/evaluation/" # train model output folder
config.log_name = config.model_name + "_" + config.training_desc

# transform image
config.image_size = 224
config.interpolation_type = "bicubic"
if config.model_name == "clip":
    config.normalize_type = "clip"
if config.model_name == "dinov2" or config.model_name == "baseline":
    config.normalize_type = "imagenet"

# Validation set
# "lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"
# RFW: "African_test", "Asian_test", "Caucasian_test", "Indian_test"
config.eval_path = "validation_path"
config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.batch_size_eval = 1024
