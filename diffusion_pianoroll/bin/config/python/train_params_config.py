from dataclasses import dataclass

from optimizer_config import OptimizerConfig

@dataclass
class TrainParamsConfig():
    train_max_steps: int = 50000
    save_interval_steps: int = 2000
    eval_interval_steps: int = 1000
    log_interval_steps: int = 2000
    resume: bool = False

    # Optimizer and scheduler setting
    optimizer: OptimizerConfig = OptimizerConfig()

    # Sampling
    sample_grid: list = [8, 8]
    save_array_samples: bool = True
    save_image_samples: bool = True
    save_pianoroll_samples: bool = True
