# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from typing import Optional, Union
from operator import attrgetter
from copy import deepcopy

from pypianoroll import Multitrack, StandardTrack, BinaryTrack
import pretty_midi
from pretty_midi import Instrument, PrettyMIDI
import numpy as np
import scipy.stats


DEFAULT_TEMPO = 100


def multitrack_to_pretty_midi(
    multitrack: Multitrack,
    default_tempo: Optional[float] = None,
    default_velocity: Union[int, None] = 80,
) -> PrettyMIDI:
    """Return a Multitrack object as a PrettyMIDI object.

    Args:
        default_tempo(int):
            Default tempo to use. Defaults to the first element of
            attribute `tempo`.
        default_velocity(int, None]):
            Default velocity to assign to binarized tracks. Defaults to 16.
        default_velocity(Union[int, None]):
            Default velocity to assign to binarized tracks. Defaults to 80. If
            this argument is not passed, values in pianoroll are used
            as velocity for each note.

    Returns:
        pretty_midi.PrettyMIDI: Converted PrettyMIDI object.

    Notes
    -----
    - Tempo changes are not supported.
    - Time signature changes are not supported.
    - The velocities of the converted piano rolls will be clipped to
        [0, 127].
    - Adjacent nonzero values of the same pitch will be considered
        a single note with their mean as its velocity.

    """
    if default_tempo is not None:
        tempo = default_tempo
    elif multitrack.tempo is not None:
        tempo = float(scipy.stats.hmean(multitrack.tempo))
    else:
        tempo = DEFAULT_TEMPO

    if default_tempo is not None:
        calc_velocities = lambda _: default_tempo  # NOQA
    else:
        calc_velocities = lambda x: np.mean(x)  # NOQA

    # Create a PrettyMIDI instance
    midi = PrettyMIDI(initial_tempo=tempo)

    # Compute length of a time step
    time_step_length = 60.0 / tempo / multitrack.resolution

    for track in multitrack.tracks:
        instrument = Instrument(
            program=track.program, is_drum=track.is_drum, name=track.name
        )
        track = track.standardize()
        if isinstance(track, BinaryTrack):
            processed = track.set_nonzeros(default_velocity)
        elif isinstance(track, StandardTrack):
            copied = deepcopy(track)
            processed = copied.clip()
        else:
            raise ValueError(
                f"Expect BinaryTrack or StandardTrack, but got {type(track)}."
            )
        clipped = processed.pianoroll.astype(np.uint8)
        binarized = clipped > 0
        padded = np.pad(binarized, ((1, 1), (0, 0)), "constant")
        diff = np.diff(padded.astype(np.int8), axis=0)

        positives = np.nonzero((diff > 0).T)
        pitches = positives[0]
        note_ons = positives[1]
        note_on_times = time_step_length * note_ons
        note_offs = np.nonzero((diff < 0).T)[1]
        note_off_times = time_step_length * note_offs

        for idx, pitch in enumerate(pitches):
            velocity = calc_velocities(
                clipped[note_ons[idx]: note_offs[idx], pitch])
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=note_on_times[idx],
                end=note_off_times[idx],
            )
            instrument.notes.append(note)

        instrument.notes.sort(key=attrgetter("start"))
        midi.instruments.append(instrument)

    return midi


# NOTE: This function may not be implemented correctly
def insert_blank_between_samples(
        pianoroll: np.ndarray,
        sample_resolution: int,
        measure_resolution: int) -> np.ndarray:
    """insert blank measure between each samples
    **This function may not be implemented correctly**

    Args:
        pianoroll (np.ndarray): pianoroll. shape=(track, timestep, pitch)
        sample_resolution (int): timestep per sample
        measure_resolution (int): timestep per measure

    Returns:
        np.ndarray: new pianoroll
    """
    samples = pianoroll.shape[1] // sample_resolution
    blank_measure = np.zeros(
        (pianoroll.shape[0], measure_resolution, pianoroll.shape[2]))
    for i in range(samples - 1):
        # insert blank measure from tail to head
        target = (samples - 1 - i) * sample_resolution
        pianoroll = np.concatenate((
            pianoroll[:, :target, :],
            blank_measure,
            pianoroll[:, target:, :]), axis=1)
    return pianoroll
