from dataclasses import dataclass

from model_config import ModelConfig
from data_config import DataConfig


@dataclass
class TrainConfig:
    out_dir: str = "out/trial"
    seed: int = 12345
    sf2_path: str = "./font.sf2"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
