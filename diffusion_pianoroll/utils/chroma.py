import numpy as np


def to_chroma(pianoroll: np.ndarray) -> np.ndarray:
    """convert pianoroll into chroma feature

    Args:
        pianoroll (np.ndarray): pianoroll.
            shape = (n_tracks, n_timestep, n_pitch)

    Returns:
        np.ndarray: chroma feature. shape = (n_tracks, n_timestep, 12)
    """

    reminder = pianoroll.shape[2] % 12
    if (reminder != 0):
        pianoroll = np.pad(pianoroll, ((0, 0), (0, 0), (0, 12 - reminder)))
    reshaped = pianoroll.reshape(
        pianoroll.shape[0],
        pianoroll.shape[1],
        pianoroll.shape[2] // 12 + int(reminder > 0),  # octave
        12,
    )
    return np.sum(reshaped, 2)
