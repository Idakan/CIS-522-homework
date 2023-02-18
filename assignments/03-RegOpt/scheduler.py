from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        if self.last_epoch == 0:
            return [x["lr"] for x in self.optimizer.param_groups]

        else:
            return [x["lr"] * self.gamma for x in self.optimizer.param_groups]


class CustomLRScheduler2(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        super(CustomLRScheduler2, self).__init__(optimizer, last_epoch)
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        self.base_lrs = self.optimizer.step()
        return [i for i in self.base_lrs]
