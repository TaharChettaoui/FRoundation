import os
import sys
import logging
import time
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from datetime import datetime

def init_logging(log_root, rank, models_root, logfile=None):
    if rank == 0:
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        file_name = "training.log" if logfile is None else logfile
        
        handler_file = logging.FileHandler(os.path.join(models_root, file_name))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)

        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
        log_root.info('Number of GPUs: %d' %  dist.get_world_size())


class TrainingLogger():
    def __init__(self, rank, output_path):
        if not os.path.exists(output_path) and rank == 0:
            os.makedirs(output_path)

        self.log_root = logging.getLogger()
        init_logging(self.log_root, rank, output_path)

    def pre_training(self, trainset, total_step, config):
        logging.info("Trainset lenght: %d" % len(trainset))
        logging.info("Total Step is: %d" % total_step)
        logging.info("Config is: {}".format(config.__dict__))


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss.avg, epoch, global_step, time_for_end
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, save_every, output="./"):
        self.rank: int = rank
        self.output: str = output
        self.save_every: int = save_every

    def __call__(self, epoch, backbone: torch.nn.Module, header: torch.nn.Module = None):
        if self.rank == 0 and epoch > 1 and (epoch+1) % self.save_every == 0:
            torch.save(backbone.module.state_dict(), os.path.join(self.output, str(epoch+1) + "_backbone.pth"))
        if header is not None and self.rank == 0 and epoch > 1 and (epoch+1) % self.save_every == 0:
            torch.save(header.module.state_dict(), os.path.join(self.output, str(epoch+1) + "_header.pth"))


class CallBackTensorboard():
    def __init__(self, rank, config):
        log_dir = config.output_path + "/tensorboard"
        self.writer = SummaryWriter(log_dir)
        self.log_every = config.log_every
        self.eval_every = config.eval_every
        self.rank = rank
        self.config = config

    def log_hyperparameters(self):
        param_dict = {
            key: str(value) if isinstance(value, list) else value
            for key, value in self.config.__dict__.items()
        }

        metric_dict = {"training/loss": 0}
        run_name = 'hparam'

        self.writer.add_hparams(
            hparam_dict=param_dict,
            metric_dict=metric_dict,
            run_name=run_name
        )

    def log_info(self, global_step, loss, learning_rate, model):
        if self.rank == 0 and global_step > 0 and global_step % self.log_every  == 0:
            self.writer.add_scalar('training/loss', loss, global_step)
            self.writer.add_scalar('training/learning_rate', learning_rate, global_step)

            # gradient norm
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('training/gradient_norm', total_norm, global_step)

    def log_on_epoch_end(self, epoch, model):
        if self.rank == 0:
            # Weights histogram
            for name, param in model.named_parameters():
                if param.requires_grad:
                    hist_name = "grad/" + name
                    self.writer.add_histogram(hist_name, param.data, epoch)

    def log_verificiation(self, epoch, results_dict):
        if results_dict != None and self.rank == 0 and epoch > 0 and (epoch+1) % self.eval_every == 0:
            for key, value in results_dict.items():
                desc = "validation/" + str(key)
                self.writer.add_scalar(desc, value, epoch)
        
            # log average values
            average_val = sum(results_dict.values()) / len(results_dict)
            self.writer.add_scalar("validation/average", average_val, epoch)

    def close(self):
        self.writer.close()
    