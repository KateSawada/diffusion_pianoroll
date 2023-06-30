from dataclasses import dataclass

from diffusion_pianoroll.bin.config.python.model_config import ModelConfig
from diffusion_pianoroll.bin.config.python.data_config import DataConfig
from diffusion_pianoroll.bin.config.python.train_params_config \
    import TrainParamsConfig


@dataclass
class TrainConfig:
    out_dir: str = "out/trial"
    seed: int = 12345
    sf2_path: str = "./font.sf2"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train_params: TrainParamsConfig = TrainParamsConfig()
