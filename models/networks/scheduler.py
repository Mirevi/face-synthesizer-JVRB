from enum import Enum

from torch.optim import lr_scheduler

from config import BaseConfig, ConfigOptionPackage, ConfigOptionMetadata


class SchedulerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(LRPolicy, 'lr_policy', LRPolicy.linear, 'learning rate policy.',
                                 choices=list(LRPolicy)),
        ]

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        if options.lr_policy is LRPolicy.step:
            metadata.extend([ConfigOptionMetadata(int, 'lr_decay_iters', 50,
                                                  'multiply by a gamma every lr_decay_iters iterations')])
        return metadata


class LRPolicy(Enum):
    linear = 'linear'
    step = 'step'
    plateau = 'plateau'
    cosine = 'cosine'

    def __str__(self):
        return self.value


def get_scheduler(config: BaseConfig, optimizer, initial_epoch):
    """Return a learning rate scheduler

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    # validate config
    if SchedulerCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    from config.train_config import TrainCOP
    if TrainCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    if config['lr_policy'] == LRPolicy.linear:
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + initial_epoch - config['n_epochs']) / float(config['n_epochs_decay'] + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config['lr_policy'] == LRPolicy.step:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_iters'], gamma=0.1)
    elif config['lr_policy'] == LRPolicy.plateau:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config['lr_policy'] == LRPolicy.cosine:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_epochs'], eta_min=0)
    else:
        return NotImplementedError('learning rate policy {} is not implemented'.format(config['lr_policy']))

    return scheduler
