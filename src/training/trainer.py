import torch
import logging
import clip

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from utils.logging import CallBackLogging, AverageMeter, CallBackModelCheckpoint, CallBackTensorboard
from utils.evaluation import CallBackVerification
from .scheduler import get_scheduler
from utils.utils import print_trainable_parameters

########  Default Trainer ########
class Trainer():
    def __init__(self, rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header=None):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.header = header
        self.transform = transform
        self.trainset = trainset
        self.dataloader = dataloader
        self.train_sampler = train_sampler
        self.training_type = training_type
        self.config = config

        self.start_epoch = 0
        self.global_step = self.config.global_step
        self.total_step = int(len(self.trainset) / config.batch_size / self.world_size * config.num_epoch)

        # Callback
        self.callback_logging = CallBackLogging(
            config.log_every, rank, self.total_step, config.batch_size, world_size, writer=None
        )
        self.callback_verification = CallBackVerification(
            config.eval_every , rank, config.val_targets, config.eval_path, config.image_size, 
            self.transform, config.batch_size_eval, config.model_name
        )
        self.callback_save_model = CallBackModelCheckpoint(rank, config.save_every, output=config.output_path)
        self.tensorboard_callback = CallBackTensorboard(rank, self.config)
        self.tensorboard_callback.log_hyperparameters()

        # Logging
        self.loss_log = AverageMeter()
        logging.info("Trainset lenght: %d" % len(self.trainset))
        logging.info("Total Step is: %d" % self.total_step)
        logging.info("Config is: {}".format(self.config.__dict__))


######################## 
########  CLIP  ########
######################## 
class TrainerClip(Trainer):
    def __init__(self, rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header):
        super().__init__(rank, world_size, model, transform, trainset, dataloader, train_sampler, training_type, config, header)

    def start_training(self):
        if self.training_type == "text_image_header":
            self.text_image_header_training()
        elif self.training_type == "image_encoder_only":
            self.image_encoder_only_training()
        else:
            raise ValueError()

    # Train clip image and text encoder with same header
    def text_image_header_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler( 
                scheduler_type=self.config.scheduler_type, 
                optimizer_model=optimizer_model, 
                epoch=self.config.num_epoch, 
                warmup=self.config.warmup, 
                num_warmup_epochs=self.config.num_warmup_epochs, 
                T_0=self.config.T_0, 
                T_mult=self.config.T_mult, 
                eta_min=self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler( 
                scheduler_type=self.config.scheduler_type, 
                optimizer_model=optimizer_header, 
                epoch=self.config.num_epoch, 
                warmup=self.config.warmup, 
                num_warmup_epochs=self.config.num_warmup_epochs, 
                T_0=self.config.T_0, 
                T_mult=self.config.T_mult, 
                eta_min=self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )
        
        # Criterion 
        criterion = torch.nn.CrossEntropyLoss()

        template = 'a photo of a {}.'
        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                texts =  [template.format(classname) for classname in target]
                texts = clip.tokenize(texts).to(self.rank)

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # text 
                features_text = self.model.module.encode_text(texts)
                if self.config.loss == "AdaFace":
                    norm_text = torch.norm(features_text, 2, 1, True)
                    output_text = torch.div(features_text, norm_text)
                    thetas_text = self.header(output_text, norm_text, target)
                else:
                    thetas_text = self.header(F.normalize(features_text), target)
                loss_text = criterion(thetas_text, target)

                # image 
                features_image = self.model.module.encode_image(images)
                if self.config.loss == "AdaFace":
                    norm_image = torch.norm(features_image, 2, 1, True)
                    output_image = torch.div(features_image, norm_image)
                    thetas_image = self.header(output_image, norm_image, target)
                else:
                    thetas_image = self.header(F.normalize(features_image), target)
                loss_image = criterion(thetas_image, target)

                # loss
                total_loss = (loss_image + loss_text) / 2
                total_loss.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(total_loss.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step, 
                    loss=total_loss.item(), 
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)
                
                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

            val_results = self.callback_verification(epoch, self.model)
            self.tensorboard_callback.log_verificiation(epoch, val_results)
            self.tensorboard_callback.log_on_epoch_end(epoch, self.model)
            self.callback_save_model(epoch, self.model)

        self.tensorboard_callback.close()


    # Train clip image and text encoder with same header
    def image_encoder_only_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler( 
                scheduler_type=self.config.scheduler_type, 
                optimizer_model=optimizer_model, 
                epoch=self.config.num_epoch, 
                warmup=self.config.warmup, 
                num_warmup_epochs=self.config.num_warmup_epochs, 
                T_0=self.config.T_0, 
                T_mult=self.config.T_mult, 
                eta_min=self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler( 
                scheduler_type=self.config.scheduler_type, 
                optimizer_model=optimizer_header, 
                epoch=self.config.num_epoch, 
                warmup=self.config.warmup, 
                num_warmup_epochs=self.config.num_warmup_epochs, 
                T_0=self.config.T_0, 
                T_mult=self.config.T_mult, 
                eta_min=self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )
        
        # Criterion 
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # image 
                features_image = self.model.module.encode_image(images)
                if self.config.loss == "AdaFace":
                    norm_image = torch.norm(features_image, 2, 1, True)
                    output_image = torch.div(features_image, norm_image)
                    thetas_image = self.header(output_image, norm_image, target)
                else:
                    thetas_image = self.header(F.normalize(features_image), target)
                loss_image = criterion(thetas_image, target)

                loss_image.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step, 
                    loss=loss_image.item(), 
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model.module.visual
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

            val_results = self.callback_verification(epoch, self.model)
            self.tensorboard_callback.log_verificiation(epoch, val_results)
            self.tensorboard_callback.log_on_epoch_end(epoch, self.model.module.visual)
            self.callback_save_model(epoch, self.model)
        
        self.tensorboard_callback.close()
