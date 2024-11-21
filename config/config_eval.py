from easydict import EasyDict as edict

config = edict()
config.model_path = None

# Model
config.model_name = "clip"
config.training_desc = "test_eval"
if config.model_name == "clip":
    config.backbone_size = "ViT-B/16"  #"ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    config.training_type = "image_encoder_only" # "text_image_header", "text_image_contrastive", "image_encoder_only", "text_image_kernel_id"

# LoRA
config.use_lora = False
config.lora_r = 16 # 2, 4, 8, 16
config.lora_a = 32 # 2 - 512
config.lora_dropout = 0.25
config.lora_bias = "none"  #  ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’,
config.rslora = True
if config.model_name == "clip":
    config.lora_target_modules = ['q', 'v'] # ['q', 'k', 'v', 'o']

# logging
config.output = "output/evaluation/" # train model output folder
config.log_name = config.model_name + "_" + config.training_desc

# transform image
config.image_size = 224 
if config.model_name == "clip":
    config.normalize_type = "clip"
    config.interpolation_type = "bicubic"

# Validation set
# "lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"
# RFW: "African_test", "Asian_test", "Caucasian_test", "Indian_test"
config.eval_path = "validation_path" # "/data/Biometrics/validation", "/home/chettaou/data/validation"
config.val_targets = ["lfw"] # , "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw", "African_test", "Asian_test", "Caucasian_test", "Indian_test"]
config.batch_size_eval = 1024
