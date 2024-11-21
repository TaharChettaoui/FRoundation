import logging
import os
import sys
import torch
import argparse
import torch.distributed as dist

from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.getcwd()))

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image
from utils.evaluation import CallBackVerification
from utils.logging import init_logging
from finetuning import apply_lora_model


def evaluate(args):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    log_root = logging.getLogger()
    if not os.path.exists(cfg.output) and local_rank == 0:
        os.makedirs(cfg.output)
    init_logging(log_root, local_rank, cfg.output, logfile=cfg.log_name)

    # Transform
    transform = transform_image(
        image_size=cfg.image_size, 
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    callback_verification = CallBackVerification(
        5, local_rank, cfg.val_targets, cfg.eval_path, 
        cfg.image_size, transform, cfg.batch_size_eval, cfg.model_name
    )

    model = get_model(local_rank, **cfg)

    # LoRA
    if cfg.use_lora:
        apply_lora_model(local_rank, model, **cfg)

    # Load trained model
    if cfg.model_path:
        print("Loading model from path: " + cfg.model_path)
        model.backbone.load_state_dict(torch.load(cfg.model_path))

    model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    model.eval()
    
    # Eval
    callback_verification(4, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation job')
    parser.add_argument('--local-rank', type=int, help='local_rank')
    parser.add_argument('--debug', default=False, type=bool, help='Log additional debug informations')
    args = parser.parse_args()

    evaluate(args)
