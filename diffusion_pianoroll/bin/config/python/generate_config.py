from dataclasses import dataclass

from diffusion_pianoroll.bin.config.python.model_config import ModelConfig
from diffusion_pianoroll.bin.config.python.data_config import DataConfig


@dataclass
class GenerateConfig:
    out_dir: str = "out/trial"
    steps: int = 50000
    seed: int = 12345
    sf2_path: str = "./font.sf2"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    checkpoint_path: str = ""
    n_samples: int = 4
    noise_timesteps: int = 200
    beta_schedule: str = "squaredcos_cap_v2"
