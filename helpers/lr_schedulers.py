import torch


class LinearScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    A custom linear scheduler that anneals the learning rate after a certain number of iterations.
    """

    def __init__(
        self,
        optimizer,
        start_epoch,
        decay_length,
        min_lr=1e-7,
        warmup_length=500,
        num_processes=1,
        last_epoch=-1,
        **kwargs,
    ):
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        self.decay_length = decay_length
        self.warmup_length = warmup_length
        self.num_processes = num_processes

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for each parameter group.

        Returns:
            list[float]: The learning rates for each parameter group.
        """
        epoch = self.last_epoch / self.num_processes
        if epoch < self.warmup_length:
            gamma = epoch / self.warmup_length
            return [base_lr * gamma + self.min_lr * (1 - gamma) for base_lr in self.base_lrs]
        if epoch > self.start_epoch:
            gamma = 1 - (epoch - self.start_epoch) / self.decay_length
            return [
                max(self.min_lr, base_lr * gamma + self.min_lr * (1 - gamma)) for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


class ConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    A constant scheduler that keeps the same learning rate.
    """

    def __init__(
        self,
        optimizer,
        start_epoch,
        decay_length,
        min_lr=1e-7,
        warmup_length=500,
        num_processes=1,
        last_epoch=-1,
        **kwargs,
    ):
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        self.decay_length = decay_length
        self.warmup_length = warmup_length
        self.num_processes = num_processes

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for each parameter group.

        Returns:
            list[float]: The learning rates for each parameter group.
        """
        return self.base_lrs


def get_lr_scheduler(
    name,
):
    if name.lower() in ["constant", "null"]:
        return ConstantScheduler
    elif name.lower() in ["linear"]:
        return LinearScheduler
