import torch


def lr_step_func(epoch, func_drop=[22, 30, 40]):
    return  0.1 ** len([m for m in func_drop if m - 1 <= epoch])


def get_scheduler(scheduler_type, optimizer_model, epoch, warmup, num_warmup_epochs, T_0, T_mult, eta_min, lr_func_drop, warmup_factor=1):
    lr_func = lambda epoch: lr_step_func(epoch, func_drop=lr_func_drop)

    if scheduler_type == "lambda":
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer_model, 
            lr_lambda=lr_func
        )
    elif scheduler_type == "cosine":
        scheduler_model = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer_model, 
            T_0=T_0, 
            T_mult=T_mult, 
            eta_min=eta_min
        )
    else:
        raise ValueError()
        
    if warmup == True:
        scheduler_warmup_model = torch.optim.lr_scheduler.ConstantLR(
            optimizer_model, 
            factor=warmup_factor, 
            total_iters=num_warmup_epochs
        )

        scheduler_model = torch.optim.lr_scheduler.SequentialLR(
            optimizer_model, 
            schedulers=[scheduler_warmup_model, scheduler_model], 
            milestones=[num_warmup_epochs]
        )

    return scheduler_model
