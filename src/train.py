import os
import sys
import argparse
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import destroy_process_group
from datetime import timedelta

sys.path.append(os.path.join(os.getcwd()))

from config.config import config as cfg
from backbone import get_model, get_output_dim
from training import get_header, get_trainer
from data import get_dataset
from data.loaders import DataLoaderX
from data.transform import transform_image
from finetuning import apply_lora_model
from utils.logging import TrainingLogger

torch.backends.cudnn.benchmark = True

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout


def main(args):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) # args.local_rank
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    cfg.batch_size = cfg.global_batch_size // world_size # batch_size per GPU

    # Logging
    TrainingLogger(local_rank, cfg.output_path)

    # Transform init
    transform = transform_image(
        image_size=cfg.image_size, 
        normalize_type=cfg.normalize_type,
        horizontal_flip=cfg.horizontal_flip,
        rand_augment=cfg.rand_augment,
        interpolation_type=cfg.interpolation_type
    )
    transform_val = transform_image(
        image_size=cfg.image_size, 
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    # Dataset
    trainset = get_dataset(local_rank, transform, **cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    dataloader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size, 
        pin_memory=True, drop_last=True, num_workers=0, sampler=train_sampler
    )

    # Model
    model = get_model(local_rank, **cfg)
    if cfg.use_lora: # LoRA
        apply_lora_model(local_rank, model, **cfg)
    model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    model.train()

    # Header
    output_dim = get_output_dim(**cfg)
    header = get_header(rank=local_rank, backbone_out_dim=output_dim, **cfg).to(local_rank)
    header = DistributedDataParallel(module=header, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    header.train()

    # Training 
    model_trainer = get_trainer(
        rank=local_rank,     
        world_size=world_size,
        model_name=cfg.model_name,
        model=model, 
        transform=transform_val, 
        trainset=trainset, 
        dataloader=dataloader, 
        train_sampler=train_sampler,
        training_type=cfg.training_type,
        config=cfg,
        header=header
    )
    model_trainer.start_training()

    if local_rank == 0:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('--local-rank', type=int, help='local_rank')
    parser.add_argument('--mode', default='training', choices=['training', 'evaluation'], help='train or eval mode')
    parser.add_argument('--debug', default=False, type=bool, help='Log additional debug informations')
    args = parser.parse_args()

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    main(args)
