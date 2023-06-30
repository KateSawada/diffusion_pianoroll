from dataclasses import dataclass

import torch
import torch.optim.optimizer


@dataclass
class OptimizerConfig:
    _target_: torch.optim.optimizer.Optimizer
