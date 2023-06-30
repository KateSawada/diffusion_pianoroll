from dataclasses import dataclass, field


@dataclass
class DataConfig:
    train_pianoroll: str = "data/lpd/data/lpd_train.txt"
    valid_pianoroll: str = "data/lpd/data/lpd_valid.txt"
    eval_pianoroll: str = "data/lpd/data/lpd_eval.txt"
    allow_cache: bool = False

    n_tracks: int = 5
    measure_resolution: int = 48
    beat_resolution: int = 12
    n_pitches: int = 84
    n_measures: int = 4
    lowest_pitch: int = 24

    # Data loader setting
    batch_size: int = 64
    num_workers: int = 1
    pin_memory: bool = True

    is_drums: list[bool] = field(default_factory=lambda: [
        True,
        False,
        False,
        False,
        False,
    ])
    programs: list[int] = field(default_factory=lambda: [
        0,
        0,
        25,
        33,
        48,
    ])
    tempo: int = 100
    track_names: list[str] = field(default_factory=lambda: [
        "Drums",
        "Piano",
        "Guitar",
        "Bass",
        "Strings",
    ])

    colormap: list[list[int]] = field(default_factory=lambda: [
        [1., 0., 0.],
        [1., .5, 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., .5, 1.],
    ])
