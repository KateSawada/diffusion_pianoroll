from dataclasses import dataclass

import torch
from torch.optim.optimizer import Optimizer


@dataclass
class OptimizerConfig:
    _target_: Optimizer = torch.optim.Adam
