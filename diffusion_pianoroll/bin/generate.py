from collections import defaultdict
import os
import sys
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMScheduler
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pypianoroll
from pypianoroll import Multitrack, Track

import diffusion_pianoroll.bin.config.python as config_dataclass
from diffusion_pianoroll.datasets import PianorollDataset
from diffusion_pianoroll.models import MuseGANAutoEncoder
from diffusion_pianoroll.utils import midi
from diffusion_pianoroll.utils import utils

logger = getLogger(__name__)


def save(
        pianoroll_tensor: torch.Tensor,
        config: config_dataclass.GenerateConfig
):
    measure_resolution = config.data.measure_resolution
    tempo_array = np.full((4 * 4 * measure_resolution, 1), config.data.tempo)

    pianoroll = pianoroll_tensor.cpu().detach().numpy().copy()

    pianoroll = pianoroll.reshape(
        config.data.n_tracks, -1, config.data.n_pitches)

    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config.data.programs, config.data.is_drums, config.data.track_names)
    ):
        if len(pianoroll[idx]) >= measure_resolution * 4 * 4:
            pianoroll_ = np.pad(
                # plot 4 samples
                pianoroll[idx, :measure_resolution * 4 * 4] > 0.5,
                ((0, 0), (
                    config.data.lowest_pitch,
                    128 - config.data.lowest_pitch - config.data.n_pitches))
            )
        else:
            pianoroll_ = np.pad(
                pianoroll[idx] > 0.5,
                ((0, 0), (
                    config.data.lowest_pitch,
                    128 - config.data.lowest_pitch - config.data.n_pitches))
            )
        tracks.append(
            Track(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll_
            )
        )
    m = Multitrack(
        tracks=tracks,
        tempo=tempo_array,
        resolution=config.data.beat_resolution)

    # save pianoroll as png
    axs = m.plot()

    for ax in axs:
        for x in range(
            measure_resolution,
            4 * measure_resolution * config.data.n_measures,
            measure_resolution
        ):
            if x % (measure_resolution * 4) == 0:
                ax.axvline(x - 0.5, color='k')
            else:
                ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
    plt.gcf().set_size_inches((16, 8))

    utils.makedirs_if_not_exists(os.path.join(config.out_dir, "pianoroll"))
    plt.savefig(os.path.join(
        config.out_dir, "pianoroll", f"pianoroll-{config.steps}steps.png"))
    plt.clf()
    plt.close()

    # save npy
    # self.mkdir(os.path.join(config.out_dir, "npy"))
    # np.save(os.path.join(config.out_dir, "npy", f"npy-{self.steps}steps.npy"), pianoroll)

    # midi npyのsample間に1小節の空白をあける pianoroll.shape = (tracks, timestep, pitches)
    # pianoroll_blank = midi.insert_blank_between_samples(pianoroll, config.data.measure_resolution * config.data.n_tracks , config.data.measure_resolution)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config.data.programs, config.data.is_drums, config.data.track_names)
    ):
        pianoroll_ = np.pad(
            pianoroll[idx] > 0.5,
            ((0, 0), (
                config.data.lowest_pitch,
                128 - config.data.lowest_pitch - config.data.n_pitches))
        )
        tracks.append(
            Track(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll_
            )
        )
    m = Multitrack(
        tracks=tracks,
        tempo=tempo_array,
        resolution=config.data.beat_resolution)
    mid = midi.multitrack_to_pretty_midi(m)

    utils.makedirs_if_not_exists(os.path.join(config.out_dir, "mid"))
    mid_path = os.path.join(
        config.out_dir, "mid", f"mid-{config.steps}steps.mid")
    mid.write(mid_path)


@hydra.main(version_base=None, config_path="config/yaml",
            config_name="generate")
def main(config: config_dataclass.GenerateConfig) -> None:
    """Run training process."""
    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936  # NOQA
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    # load pre-trained model from checkpoint file
    state_dict = torch.load(to_absolute_path(config.checkpoint_path), map_location="cpu")
    # TODO: load from config
    model = MuseGANAutoEncoder(
        config.data.n_tracks,
        config.data.n_measures,
        config.data.measure_resolution,
        config.data.measure_resolution // config.data.beat_resolution,
        config.data.n_pitches,
        config.model.d_latent,
    )
    model.load_state_dict(state_dict["model"]["diffusion"])
    model.eval().to(device)

    sample = torch.randn(
        config.n_samples,
        config.data.n_tracks,
        config.data.n_measures,
        config.data.measure_resolution,
        config.data.n_pitches,
    ).to(device)
    # (batch, 5, 4, 48, 84)

    # noise scheduler
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=config.noise_timesteps,
        beta_schedule=config.beta_schedule,
    )
    for i, t in enumerate(ddpm_scheduler.timesteps):

        # Get model pred
        with torch.no_grad():
            residual = model(sample, t)[0]

        # Update sample with step
        sample = ddpm_scheduler.step(residual, t, sample).prev_sample

    save(sample, config)


if __name__ == "__main__":
    main()
