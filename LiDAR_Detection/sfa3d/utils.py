import os
import logging
import torch
from torch.optim import SGD, lr_scheduler
import numpy as np
import time
import torch.distributed as dist
import copy
import math
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import types
class Logger():
    """
        Create logger to save logs during training
        Args:
            logs_dir:
            saved_fn:

        Returns:

        """

    def __init__(self, logs_dir, saved_fn):
        logger_fn = 'logger_{}.txt'.format(saved_fn)
        logger_path = os.path.join(logs_dir, logger_fn)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # formatter = logging.Formatter('%(asctime)s:File %(module)s.py:Func %(funcName)s:Line %(lineno)d:%(levelname)s: %(message)s')
        formatter = logging.Formatter(
            '%(asctime)s: %(module)s.py - %(funcName)s(), at Line %(lineno)d:%(levelname)s:\n%(message)s')

        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)

class lr_schedulers():
    class _LRMomentumScheduler(lr_scheduler._LRScheduler):
        def __init__(self, optimizer, last_epoch=-1):
            if last_epoch == -1:
                for group in optimizer.param_groups:
                    group.setdefault('initial_momentum', group['momentum'])
            else:
                for i, group in enumerate(optimizer.param_groups):
                    if 'initial_momentum' not in group:
                        raise KeyError("param 'initial_momentum' is not specified "
                                       "in param_groups[{}] when resuming an optimizer".format(i))
            self.base_momentums = list(map(lambda group: group['initial_momentum'], optimizer.param_groups))
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            raise NotImplementedError

        def get_momentum(self):
            raise NotImplementedError

        def step(self, epoch=None):
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            for param_group, lr, momentum in zip(self.optimizer.param_groups, self.get_lr(), self.get_momentum()):
                param_group['lr'] = lr
                param_group['momentum'] = momentum

    class ParameterUpdate(object):
        """A callable class used to define an arbitrary schedule defined by a list.
        This object is designed to be passed to the LambdaLR or LambdaScheduler scheduler to apply
        the given schedule.

        Arguments:
            params {list or numpy.array} -- List or numpy array defining parameter schedule.
            base_param {float} -- Parameter value used to initialize the optimizer.
        """

        def __init__(self, params, base_param):
            self.params = np.hstack([params, 0])
            self.base_param = base_param

        def __call__(self, epoch):
            return self.params[epoch] / self.base_param

    def apply_lambda(last_epoch, bases, lambdas):
        return [base * lmbda(last_epoch) for lmbda, base in zip(lambdas, bases)]

    class LambdaScheduler(_LRMomentumScheduler):
        """Sets the learning rate and momentum of each parameter group to the initial lr and momentum
        times a given function. When last_epoch=-1, sets initial lr and momentum to the optimizer
        values.
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            lr_lambda (function or list): A function which computes a multiplicative
                factor given an integer parameter epoch, or a list of such
                functions, one for each group in optimizer.param_groups.
                Default: lambda x:x.
            momentum_lambda (function or list): As for lr_lambda but applied to momentum.
                Default: lambda x:x.
            last_epoch (int): The index of last epoch. Default: -1.
        Example:
            # >>> # Assuming optimizer has two groups.
            # >>> lr_lambda = [
            # ...     lambda epoch: epoch // 30,
            # ...     lambda epoch: 0.95 ** epoch
            # ... ]
            # >>> mom_lambda = [
            # ...     lambda epoch: max(0, (50 - epoch) // 50),
            # ...     lambda epoch: 0.99 ** epoch
            # ... ]
            # >>> scheduler = LambdaScheduler(optimizer, lr_lambda, mom_lambda)
            # >>> for epoch in range(100):
            # >>>     train(...)
            # >>>     validate(...)
            # >>>     scheduler.step()
        """

        def __init__(self, optimizer, lr_lambda=lambda x: x, momentum_lambda=lambda x: x, last_epoch=-1):
            self.optimizer = optimizer

            if not isinstance(lr_lambda, (list, tuple)):
                self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
            else:
                if len(lr_lambda) != len(optimizer.param_groups):
                    raise ValueError("Expected {} lr_lambdas, but got {}".format(
                        len(optimizer.param_groups), len(lr_lambda)))
                self.lr_lambdas = list(lr_lambda)

            if not isinstance(momentum_lambda, (list, tuple)):
                self.momentum_lambdas = [momentum_lambda] * len(optimizer.param_groups)
            else:
                if len(momentum_lambda) != len(optimizer.param_groups):
                    raise ValueError("Expected {} momentum_lambdas, but got {}".format(
                        len(optimizer.param_groups), len(momentum_lambda)))
                self.momentum_lambdas = list(momentum_lambda)

            self.last_epoch = last_epoch
            super().__init__(optimizer, last_epoch)

        def state_dict(self):
            """Returns the state of the scheduler as a :class:`dict`.
            It contains an entry for every variable in self.__dict__ which
            is not the optimizer.
            The learning rate and momentum lambda functions will only be saved if they are
            callable objects and not if they are functions or lambdas.
            """
            state_dict = {key: value for key, value in self.__dict__.items()
                          if key not in ('optimizer', 'lr_lambdas', 'momentum_lambdas')}
            state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)
            state_dict['momentum_lambdas'] = [None] * len(self.momentum_lambdas)

            for idx, (lr_fn, mom_fn) in enumerate(zip(self.lr_lambdas, self.momentum_lambdas)):
                if not isinstance(lr_fn, types.FunctionType):
                    state_dict['lr_lambdas'][idx] = lr_fn.__dict__.copy()
                if not isinstance(mom_fn, types.FunctionType):
                    state_dict['momentum_lambdas'][idx] = mom_fn.__dict__.copy()

            return state_dict

        def load_state_dict(self, state_dict):
            """Loads the schedulers state.
            Arguments:
                state_dict (dict): scheduler state. Should be an object returned
                    from a call to :meth:`state_dict`.
            """
            lr_lambdas = state_dict.pop('lr_lambdas')
            momentum_lambdas = state_dict.pop('momentum_lambdas')
            self.__dict__.update(state_dict)

            for idx, fn in enumerate(lr_lambdas):
                if fn is not None:
                    self.lr_lambdas[idx].__dict__.update(fn)

            for idx, fn in enumerate(momentum_lambdas):
                if fn is not None:
                    self.momentum_lambdas[idx].__dict__.update(fn)

        def get_lr(self):
            return lr_schedulers.LambdaScheduler.apply_lambda(self.last_epoch, self.base_lrs, self.lr_lambdas)

        def get_momentum(self):
            return lr_schedulers.LambdaScheduler.apply_lambda(self.last_epoch, self.base_momentums, self.momentum_lambdas)

    class ParameterUpdate(object):
        """A callable class used to define an arbitrary schedule defined by a list.
        This object is designed to be passed to the LambdaLR or LambdaScheduler scheduler to apply
        the given schedule. If a base_param is zero, no updates are applied.

        Arguments:
            params {list or numpy.array} -- List or numpy array defining parameter schedule.
            base_param {float} -- Parameter value used to initialize the optimizer.
        """
        def __init__(self, params, base_param):
            self.params = np.hstack([params, 0])
            self.base_param = base_param

            if base_param < 1e-12:
                self.base_param = 1
                self.params = self.params * 0.0 + 1.0

        def __call__(self, epoch):
            return self.params[epoch] / self.base_param

    class ListScheduler(LambdaScheduler):
        """Sets the learning rate and momentum of each parameter group to values defined by lists.
        When last_epoch=-1, sets initial lr and momentum to the optimizer values. One of both of lr
        and momentum schedules may be specified.
        Note that the parameters used to initialize the optimizer are overriden by those defined by
        this scheduler.
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            lrs (list or numpy.ndarray): A list of learning rates, or a list of lists, one for each
                parameter group. One- or two-dimensional numpy arrays may also be passed.
            momentum (list or numpy.ndarray): A list of momentums, or a list of lists, one for each
                parameter group. One- or two-dimensional numpy arrays may also be passed.
            last_epoch (int): The index of last epoch. Default: -1.
        Example:
            # >>> # Assuming optimizer has two groups.
            # >>> lrs = [
            # ...     np.linspace(0.01, 0.1, 100),
            # ...     np.logspace(-2, 0, 100)
            # ... ]
            # >>> momentums = [
            # ...     np.linspace(0.85, 0.95, 100),
            # ...     np.linspace(0.8, 0.99, 100)
            # ... ]
            # >>> scheduler = ListScheduler(optimizer, lrs, momentums)
            # >>> for epoch in range(100):
            # >>>     train(...)
            # >>>     validate(...)
            # >>>     scheduler.step()
        """

        def __init__(self, optimizer, lrs=None, momentums=None, last_epoch=-1):
            groups = optimizer.param_groups
            if lrs is None:
                lr_lambda = lambda x: x
            else:
                lrs = np.array(lrs) if isinstance(lrs, (list, tuple)) else lrs
                if len(lrs.shape) == 1:
                    lr_lambda = [lr_schedulers.ParameterUpdate(lrs, g['lr']) for g in groups]
                else:
                    lr_lambda = [lr_schedulers.ParameterUpdate(l, g['lr']) for l, g in zip(lrs, groups)]

            if momentums is None:
                momentum_lambda = lambda x: x
            else:
                momentums = np.array(momentums) if isinstance(momentums, (list, tuple)) else momentums
                if len(momentums.shape) == 1:
                    momentum_lambda = [lr_schedulers.ParameterUpdate(momentums, g['momentum']) for g in groups]
                else:
                    momentum_lambda = [lr_schedulers.ParameterUpdate(l, g['momentum']) for l, g in zip(momentums, groups)]
            super().__init__(optimizer, lr_lambda, momentum_lambda)

    class RangeFinder(ListScheduler):
        """Scheduler class that implements the LR range search specified in:
            A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
            size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.

        Logarithmically spaced learning rates from 1e-7 to 1 are searched. The number of increments in
        that range is determined by 'epochs'.
        Note that the parameters used to initialize the optimizer are overriden by those defined by
        this scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            epochs (int): Number of epochs over which to run test.
        # Example:
        #     >>> scheduler = RangeFinder(optimizer, 100)
        #     >>> for epoch in range(100):
        #     >>>     train(...)
        #     >>>     validate(...)
        #     >>>     scheduler.step()
        # """

        def __init__(self, optimizer, epochs):
            lrs = np.logspace(-7, 0, epochs)
            super().__init__(optimizer, lrs)

    class OneCyclePolicy(ListScheduler):
        """Scheduler class that implements the 1cycle policy search specified in:
            A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch
            size, momentum, and weight decay. Leslie N. Smith, 2018, arXiv:1803.09820.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            lr (float or list). Maximum learning rate in range. If a list of values is passed, they
                should correspond to parameter groups.
            epochs (int): The number of epochs to use during search.
            momentum_rng (list). Optional upper and lower momentum values (may be both equal). Set to
                None to run without momentum. Default: [0.85, 0.95]. If a list of lists is passed, they
                should correspond to parameter groups.
            phase_ratio (float): Fraction of epochs used for the increasing and decreasing phase of
                the schedule. For example, if phase_ratio=0.45 and epochs=100, the learning rate will
                increase from lr/10 to lr over 45 epochs, then decrease back to lr/10 over 45 epochs,
                then decrease to lr/100 over the remaining 10 epochs. Default: 0.45.
        """

        def __init__(self, optimizer, lr, epochs, momentum_rng=[0.85, 0.95], phase_ratio=0.45):
            phase_epochs = int(phase_ratio * epochs)
            if isinstance(lr, (list, tuple)):
                lrs = [
                    np.hstack([
                        np.linspace(l * 1e-1, l, phase_epochs),
                        np.linspace(l, l * 1e-1, phase_epochs),
                        np.linspace(l * 1e-1, l * 1e-2, epochs - 2 * phase_epochs),
                    ]) for l in lr
                ]
            else:
                lrs = np.hstack([
                    np.linspace(lr * 1e-1, lr, phase_epochs),
                    np.linspace(lr, lr * 1e-1, phase_epochs),
                    np.linspace(lr * 1e-1, lr * 1e-2, epochs - 2 * phase_epochs),
                ])

            if momentum_rng is not None:
                momentum_rng = np.array(momentum_rng)
                if len(momentum_rng.shape) == 2:
                    for i, g in enumerate(optimizer.param_groups):
                        g['momentum'] = momentum_rng[i][1]
                    momentums = [
                        np.hstack([
                            np.linspace(m[1], m[0], phase_epochs),
                            np.linspace(m[0], m[1], phase_epochs),
                            np.linspace(m[1], m[1], epochs - 2 * phase_epochs),
                        ]) for m in momentum_rng
                    ]
                else:
                    for i, g in enumerate(optimizer.param_groups):
                        g['momentum'] = momentum_rng[1]
                    momentums = np.hstack([
                        np.linspace(momentum_rng[1], momentum_rng[0], phase_epochs),
                        np.linspace(momentum_rng[0], momentum_rng[1], phase_epochs),
                        np.linspace(momentum_rng[1], momentum_rng[1], epochs - 2 * phase_epochs),
                    ])
            else:
                momentums = None

            super().__init__(optimizer, lrs, momentums)

class misc:

    def make_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # or os.makedirs(folder_name, exist_ok=True)

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
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

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def display(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))

        def get_message(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            return '\t'.join(entries)

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def time_synchronized():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()

class torch_utils:
    __all__ = ['convert2cpu', 'convert2cpu_long', 'to_cpu', 'reduce_tensor', 'to_python_float', '_sigmoid']

    def convert2cpu(gpu_matrix):
        return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def convert2cpu_long(gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

    def to_cpu(tensor):
        return tensor.detach().cpu()

    def reduce_tensor(tensor, world_size):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt /= world_size
        return rt

    def to_python_float(t):
        if hasattr(t, 'item'):
            return t.item()
        else:
            return t[0]

    def _sigmoid(x):
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

class train_utils:
    def create_optimizer(configs, model):
        """Create optimizer for training process
        """
        if hasattr(model, 'module'):
            train_params = [param for param in model.module.parameters() if param.requires_grad]
        else:
            train_params = [param for param in model.parameters() if param.requires_grad]

        if configs.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(train_params, lr=configs.lr, momentum=configs.momentum, nesterov=True)
        elif configs.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(train_params, lr=configs.lr, weight_decay=configs.weight_decay)
        else:
            assert False, "Unknown optimizer type"

        return optimizer

    def create_lr_scheduler(optimizer, configs):
        """Create learning rate scheduler for training process"""

        if configs.lr_type == 'multi_step':
            def multi_step_scheduler(i):
                if i < configs.steps[0]:
                    factor = 1.
                elif i < configs.steps[1]:
                    factor = 0.1
                else:
                    factor = 0.01

                return factor

            lr_scheduler = LambdaLR(optimizer, multi_step_scheduler)

        elif configs.lr_type == 'cosin':
            # Scheduler https://arxiv.org/pdf/1812.01187.pdf
            lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
            lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        elif configs.lr_type == 'one_cycle':
            lr_scheduler = lr_schedulers.OneCyclePolicy(optimizer, configs.lr, configs.num_epochs, momentum_rng=[0.85, 0.95],
                                          phase_ratio=0.45)
        else:
            raise ValueError

        train_utils.plot_lr_scheduler(optimizer, lr_scheduler, configs.num_epochs, save_dir=configs.logs_dir,
                          lr_type=configs.lr_type)

        return lr_scheduler

    def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
        """Get the information to save with checkpoints"""
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        utils_state_dict = {
            'epoch': epoch,
            'configs': configs,
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
        }

        return model_state_dict, utils_state_dict

    def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch):
        """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
        model_save_path = os.path.join(checkpoints_dir, 'Model_{}_epoch_{}.pth'.format(saved_fn, epoch))
        utils_save_path = os.path.join(checkpoints_dir, 'Utils_{}_epoch_{}.pth'.format(saved_fn, epoch))

        torch.save(model_state_dict, model_save_path)
        torch.save(utils_state_dict, utils_save_path)

        print('save a checkpoint at {}'.format(model_save_path))

    def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir='', lr_type=''):
        # Plot LR simulating training for full num_epochs
        optimizer, scheduler = copy.copy(optimizer), copy.copy(scheduler)  # do not modify originals
        y = []
        for _ in range(num_epochs):
            scheduler.step()
            y.append(optimizer.param_groups[0]['lr'])
        plt.plot(y, '.-', label='LR')
        plt.xlabel('epoch')
        plt.ylabel('LR')
        plt.grid()
        plt.xlim(0, num_epochs)
        plt.ylim(0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'LR_{}.png'.format(lr_type)), dpi=200)

    if __name__ == '__main__':
        from easydict import EasyDict as edict
        from torchvision.models import resnet18

        configs = edict()
        configs.steps = [150, 180]
        configs.lr_type = 'one_cycle'  # multi_step, cosin, one_csycle
        configs.logs_dir = '../../logs/'
        configs.num_epochs = 50
        configs.lr = 2.25e-3
        net = resnet18()
        optimizer = torch.optim.Adam(net.parameters(), 0.0002)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        scheduler = create_lr_scheduler(optimizer, configs)
        for i in range(configs.num_epochs):
            print(i, scheduler.get_lr())
            scheduler.step()
