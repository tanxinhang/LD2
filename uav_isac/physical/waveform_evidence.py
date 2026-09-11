"""Minimal OTFS waveform-to-local-evidence calibration model.

This is an offline falsification model, not the online simulator and not a
hardware-faithful transceiver.  It closes one ideal cyclic OTFS block:

DD symbols -> ISFFT -> OFDM modulation -> fractional circular delay/Doppler
channel -> OFDM demodulation -> SFFT -> coherent DD matched statistic.

Multipath, common clutter, receiver-local clutter and complex AWGN are
explicit.  Pulse shaping, CP insufficiency, synchronization error, RF
impairments and unknown target phase are intentionally outside this first
gate and must not be inferred from its results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MinimalOTFSWaveform:
    delay_bins: int = 16
    doppler_bins: int = 8
    delta_f_hz: float = 15_625.0
    noise_variance: float = 1.0
    pilot_seed: int = 20260911

    @property
    def sample_count(self) -> int:
        return int(self.delay_bins) * int(self.doppler_bins)

    def validate(self) -> None:
        if int(self.delay_bins) < 2 or int(self.doppler_bins) < 2:
            raise ValueError("OTFS dimensions must both be at least two")
        if not np.isfinite(self.delta_f_hz) or self.delta_f_hz <= 0.0:
            raise ValueError("delta_f_hz must be finite and positive")
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be finite and positive")


@dataclass(frozen=True)
class WaveformEvidenceScenario:
    """Receiver-indexed physical paths for one target and clutter class."""

    target_delay_bin: np.ndarray
    target_doppler_bin: np.ndarray
    target_amplitude: np.ndarray
    common_clutter_delay_bin: np.ndarray
    common_clutter_doppler_bin: np.ndarray
    common_clutter_loading: np.ndarray
    common_clutter_std: float
    local_clutter_delay_bin: np.ndarray
    local_clutter_doppler_bin: np.ndarray
    local_clutter_std: np.ndarray
    target_secondary_relative_gain: complex = 0.0j
    target_secondary_delay_offset_bin: float = 0.0
    target_secondary_doppler_offset_bin: float = 0.0
    target_fluctuation_std: float = 0.0

    @property
    def receiver_count(self) -> int:
        return int(np.asarray(self.target_delay_bin).size)

    def validated(self) -> "WaveformEvidenceScenario":
        arrays = (
            "target_delay_bin",
            "target_doppler_bin",
            "target_amplitude",
            "common_clutter_delay_bin",
            "common_clutter_doppler_bin",
            "common_clutter_loading",
            "local_clutter_delay_bin",
            "local_clutter_doppler_bin",
            "local_clutter_std",
        )
        size = self.receiver_count
        if size < 1:
            raise ValueError("scenario must contain at least one receiver")
        for name in arrays:
            value = np.asarray(getattr(self, name))
            if value.shape != (size,) or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must be a finite receiver vector")
        target_amplitude = np.asarray(self.target_amplitude)
        if np.iscomplexobj(target_amplitude) and np.any(
            np.abs(target_amplitude.imag) > 0.0
        ):
            raise ValueError("target_amplitude must be real and non-negative")
        if np.any(target_amplitude.real < 0.0):
            raise ValueError("target_amplitude must be non-negative")
        if np.any(np.asarray(self.local_clutter_std) < 0.0):
            raise ValueError("local_clutter_std must be non-negative")
        scalars = (
            self.common_clutter_std,
            self.target_secondary_delay_offset_bin,
            self.target_secondary_doppler_offset_bin,
            self.target_fluctuation_std,
        )
        if any(not np.isfinite(value) for value in scalars):
            raise ValueError("scenario scalar parameters must be finite")
        if self.common_clutter_std < 0.0 or self.target_fluctuation_std < 0.0:
            raise ValueError("clutter/target fluctuation scales must be non-negative")
        if not np.isfinite(self.target_secondary_relative_gain):
            raise ValueError("secondary path gain must be finite")
        return self


def qpsk_dd_pilot(config: MinimalOTFSWaveform) -> np.ndarray:
    """Return a deterministic unit-modulus local calibration pilot."""
    config.validate()
    rng = np.random.default_rng(int(config.pilot_seed))
    phase = rng.integers(
        0, 4, size=(int(config.doppler_bins), int(config.delay_bins)))
    return np.exp(0.5j * np.pi * phase)


def otfs_modulate(dd_symbols: np.ndarray) -> np.ndarray:
    """Unitary rectangular-pulse OTFS modulation for one or more blocks."""
    values = np.asarray(dd_symbols, dtype=np.complex128)
    if values.ndim < 2 or min(values.shape[-2:]) < 1:
        raise ValueError("DD symbols must end with (N,M)")
    # ISFFT: positive Doppler transform, negative delay transform.
    time_frequency = np.fft.fft(
        np.fft.ifft(values, axis=-2, norm="ortho"),
        axis=-1,
        norm="ortho",
    )
    time_symbols = np.fft.ifft(time_frequency, axis=-1, norm="ortho")
    return time_symbols.reshape(values.shape[:-2] + (-1,))


def otfs_demodulate(
    time_samples: np.ndarray,
    *,
    doppler_bins: int,
    delay_bins: int,
) -> np.ndarray:
    """Inverse of :func:`otfs_modulate` under the ideal cyclic block model."""
    samples = np.asarray(time_samples, dtype=np.complex128)
    count = int(doppler_bins) * int(delay_bins)
    if samples.ndim < 1 or samples.shape[-1] != count:
        raise ValueError("time samples do not match the declared OTFS grid")
    time_symbols = samples.reshape(
        samples.shape[:-1] + (int(doppler_bins), int(delay_bins)))
    time_frequency = np.fft.fft(time_symbols, axis=-1, norm="ortho")
    return np.fft.ifft(
        np.fft.fft(time_frequency, axis=-2, norm="ortho"),
        axis=-1,
        norm="ortho",
    )


def apply_fractional_dd_path(
    time_samples: np.ndarray,
    *,
    delay_bin: float,
    doppler_bin: float,
    gain: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Apply one ideal cyclic fractional-delay/fractional-Doppler path.

    One delay bin equals one time sample and one Doppler bin equals one cycle
    across the complete OTFS block.  The circular model is exact only when an
    adequate cyclic extension and ideal sampling can be assumed.
    """
    samples = np.asarray(time_samples, dtype=np.complex128)
    if samples.ndim < 1 or samples.shape[-1] < 2:
        raise ValueError("time_samples must contain a non-trivial block")
    if not (
        np.isfinite(delay_bin) and np.isfinite(doppler_bin)
        and np.isfinite(gain)
    ):
        raise ValueError("path parameters must be finite")
    count = samples.shape[-1]
    signed_frequency = np.fft.fftfreq(count) * count
    delay_phase = np.exp(
        -2j * np.pi * signed_frequency * float(delay_bin) / count)
    delayed = np.fft.ifft(
        np.fft.fft(samples, axis=-1, norm="ortho")
        * delay_phase,
        axis=-1,
        norm="ortho",
    )
    sample_index = np.arange(count, dtype=np.float64)
    doppler_phase = np.exp(
        2j * np.pi * float(doppler_bin) * sample_index / count)
    return complex(gain) * delayed * doppler_phase


def dd_path_response(
    pilot_dd: np.ndarray,
    *,
    delay_bin: float,
    doppler_bin: float,
) -> np.ndarray:
    """Return the unit-gain received DD response of one physical path."""
    pilot = np.asarray(pilot_dd, dtype=np.complex128)
    time = otfs_modulate(pilot)
    received = apply_fractional_dd_path(
        time, delay_bin=delay_bin, doppler_bin=doppler_bin)
    return otfs_demodulate(
        received,
        doppler_bins=pilot.shape[-2],
        delay_bins=pilot.shape[-1],
    )


def _complex_standard_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def generate_local_evidence_trace(
    config: MinimalOTFSWaveform,
    scenario: WaveformEvidenceScenario,
    *,
    trials: int,
    hypothesis: int,
    seed: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Generate receiver-local coherent DD matched-filter statistics.

    The simulation truth is consumed only inside this offline trace generator.
    Each output column is computed from that receiver's own waveform.  No
    target truth, other-receiver statistic or future sample enters the local
    evidence extractor.
    """
    config.validate()
    scenario.validated()
    count = int(trials)
    if count < 2 or int(hypothesis) not in (0, 1) or int(batch_size) < 1:
        raise ValueError("trials, hypothesis or batch_size is invalid")
    pilot = qpsk_dd_pilot(config)
    K = scenario.receiver_count

    def responses(delay: np.ndarray, doppler: np.ndarray) -> np.ndarray:
        return np.asarray([
            dd_path_response(
                pilot,
                delay_bin=float(delay[index]),
                doppler_bin=float(doppler[index]),
            )
            for index in range(K)
        ])

    target_main = responses(
        np.asarray(scenario.target_delay_bin),
        np.asarray(scenario.target_doppler_bin),
    )
    target_secondary = responses(
        np.asarray(scenario.target_delay_bin)
        + float(scenario.target_secondary_delay_offset_bin),
        np.asarray(scenario.target_doppler_bin)
        + float(scenario.target_secondary_doppler_offset_bin),
    )
    target_response = (
        target_main
        + complex(scenario.target_secondary_relative_gain) * target_secondary
    )
    common_response = responses(
        np.asarray(scenario.common_clutter_delay_bin),
        np.asarray(scenario.common_clutter_doppler_bin),
    )
    local_response = responses(
        np.asarray(scenario.local_clutter_delay_bin),
        np.asarray(scenario.local_clutter_doppler_bin),
    )
    target_norm = np.linalg.norm(target_response.reshape(K, -1), axis=1)
    if np.any(target_norm <= 1.0e-12):
        raise RuntimeError("target matched template has zero energy")
    unit_target = target_response / target_norm[:, None, None]
    rng = np.random.default_rng(int(seed))
    output = np.empty((count, K), dtype=np.float64)
    target_amplitude = np.asarray(
        scenario.target_amplitude, dtype=np.float64)
    common_loading = np.asarray(
        scenario.common_clutter_loading, dtype=np.complex128)
    local_std = np.asarray(scenario.local_clutter_std, dtype=np.float64)

    for start in range(0, count, int(batch_size)):
        stop = min(start + int(batch_size), count)
        current = stop - start
        common_coefficient = (
            float(scenario.common_clutter_std)
            * _complex_standard_normal(rng, (current, 1))
        )
        local_coefficient = (
            _complex_standard_normal(rng, (current, K))
            * local_std[None, :]
        )
        received_dd = (
            common_coefficient[:, :, None, None]
            * common_loading[None, :, None, None]
            * common_response[None, :, :, :]
            + local_coefficient[:, :, None, None]
            * local_response[None, :, :, :]
        )
        if int(hypothesis) == 1:
            fluctuation = (
                float(scenario.target_fluctuation_std)
                * _complex_standard_normal(rng, (current, K))
            )
            received_dd = received_dd + (
                target_amplitude[None, :] + fluctuation
            )[:, :, None, None] * target_response[None, :, :, :]
        noise_time = (
            np.sqrt(float(config.noise_variance))
            * _complex_standard_normal(
                rng, (current, K, config.sample_count))
        )
        noise_dd = otfs_demodulate(
            noise_time,
            doppler_bins=int(config.doppler_bins),
            delay_bins=int(config.delay_bins),
        )
        received_dd = received_dd + noise_dd
        projection = np.sum(
            unit_target.conj()[None, :, :, :] * received_dd,
            axis=(-2, -1),
        )
        output[start:stop] = np.sqrt(2.0) * projection.real
    return output
