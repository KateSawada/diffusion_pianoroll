from dataclasses import dataclass, field

from diffusion_pianoroll.bin.config.python.optimizer_config \
    import OptimizerConfig


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
    sample_grid: list[int] = field(default_factory=lambda: [8, 8])
    save_array_samples: bool = True
    save_image_samples: bool = True
    save_pianoroll_samples: bool = True

    # noise wight schedule
    num_train_timestep: int = 200
    beta_schedule: str = "squaredcos_cap_v2"
