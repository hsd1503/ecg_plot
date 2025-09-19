#!/usr/bin/env python
"""
ECG Data Generation for Testing and Examples
Generates physiologically realistic ECG signals for development and testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


class ECGGenerator:
    """Generate synthetic ECG signals for testing and development."""

    def __init__(self, sample_rate: int = 500):
        """
        Initialize ECG generator.

        Args:
            sample_rate: Sampling frequency in Hz
        """
        self.fs = sample_rate
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def generate_normal_sinus_rhythm(self,
                                     duration: float = 10.0,
                                     heart_rate: int = 72,
                                     noise_level: float = 0.05,
                                     leads: int = 12) -> Tuple[np.ndarray, Dict]:
        """
        Generate normal sinus rhythm ECG.

        Args:
            duration: Signal duration in seconds
            heart_rate: Heart rate in beats per minute
            noise_level: Gaussian noise level (0.0 to 0.2)
            leads: Number of leads to generate

        Returns:
            Tuple of (ecg_data, metadata)
            ecg_data: Shape (leads, samples)
            metadata: Dictionary with signal parameters
        """
        n_samples = int(duration * self.fs)
        rr_interval = 60.0 / heart_rate  # seconds between beats

        # Generate basic PQRST complex template
        ecg_data = np.zeros((leads, n_samples))

        # Time axis
        t = np.linspace(0, duration, n_samples)

        # Generate beats at regular intervals
        beat_times = np.arange(0, duration, rr_interval)

        for lead_idx in range(leads):
            signal = np.zeros(n_samples)

            for beat_time in beat_times:
                if beat_time + 0.8 < duration:  # Ensure complete beat fits
                    beat_signal = self._generate_pqrst_complex(beat_time,
                                                               lead_idx,
                                                               t)
                    signal += beat_signal

            # Add physiological noise
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, n_samples)
                signal += noise

            # Add baseline wander (typical artifact)
            baseline_freq = 0.5  # Hz
            baseline_amplitude = 0.1  # mV
            baseline = baseline_amplitude * np.sin(2 * np.pi * baseline_freq * t)
            signal += baseline

            ecg_data[lead_idx] = signal

        # Apply lead-specific scaling factors
        ecg_data = self._apply_lead_characteristics(ecg_data)

        metadata = {
            'sample_rate': self.fs,
            'duration': duration,
            'heart_rate': heart_rate,
            'leads': self.lead_names[:leads],
            'signal_type': 'Normal Sinus Rhythm',
            'noise_level': noise_level
        }

        return ecg_data, metadata

    def _generate_pqrst_complex(self, beat_time: float, lead_idx: int, t: np.ndarray) -> np.ndarray:
        """Generate a single PQRST complex."""
        signal = np.zeros(len(t))

        # PQRST wave parameters (in seconds relative to R peak)
        p_time = beat_time - 0.15
        q_time = beat_time - 0.04
        r_time = beat_time
        s_time = beat_time + 0.04
        t_time = beat_time + 0.25

        # Lead-specific amplitude factors
        lead_factors = self._get_lead_amplitude_factors(lead_idx)

        # P wave (atrial depolarization)
        p_amplitude = 0.2 * lead_factors['p']
        p_width = 0.08
        signal += self._gaussian_wave(t, p_time, p_amplitude, p_width)

        # Q wave (initial ventricular depolarization)
        q_amplitude = -0.1 * lead_factors['q']
        q_width = 0.02
        signal += self._gaussian_wave(t, q_time, q_amplitude, q_width)

        # R wave (main ventricular depolarization)
        r_amplitude = 1.0 * lead_factors['r']
        r_width = 0.02
        signal += self._gaussian_wave(t, r_time, r_amplitude, r_width)

        # S wave (late ventricular depolarization)
        s_amplitude = -0.3 * lead_factors['s']
        s_width = 0.03
        signal += self._gaussian_wave(t, s_time, s_amplitude, s_width)

        # T wave (ventricular repolarization)
        t_amplitude = 0.3 * lead_factors['t']
        t_width = 0.12
        signal += self._gaussian_wave(t, t_time, t_amplitude, t_width)

        return signal

    def _gaussian_wave(self, t: np.ndarray, center: float, amplitude: float, width: float) -> np.ndarray:
        """Generate Gaussian wave centered at specific time."""
        return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

    def _get_lead_amplitude_factors(self, lead_idx: int) -> Dict[str, float]:
        """Get lead-specific amplitude factors for PQRST waves."""
        # Simplified lead characteristics based on cardiac electrical axis
        lead_factors = [
            # Limb leads
            {'p': 1.0, 'q': 0.5, 'r': 0.8, 's': 0.3, 't': 0.8},  # Lead I
            {'p': 1.2, 'q': 0.3, 'r': 1.2, 's': 0.2, 't': 1.0},  # Lead II
            {'p': 0.8, 'q': 0.2, 'r': 0.4, 's': 0.1, 't': 0.3},  # Lead III
            {'p': -0.8, 'q': -0.2, 'r': -0.6, 's': -0.1, 't': -0.5},  # aVR
            {'p': 0.6, 'q': 0.4, 'r': 0.5, 's': 0.2, 't': 0.4},  # aVL
            {'p': 1.0, 'q': 0.1, 'r': 0.7, 's': 0.1, 't': 0.6},  # aVF
            # Precordial leads
            {'p': 0.3, 'q': 0.1, 'r': 0.3, 's': 0.8, 't': -0.2},  # V1
            {'p': 0.4, 'q': 0.2, 'r': 0.5, 's': 0.6, 't': 0.1},  # V2
            {'p': 0.5, 'q': 0.3, 'r': 0.8, 's': 0.4, 't': 0.4},  # V3
            {'p': 0.6, 'q': 0.2, 'r': 1.2, 's': 0.2, 't': 0.6},  # V4
            {'p': 0.7, 'q': 0.1, 'r': 1.0, 's': 0.1, 't': 0.5},  # V5
            {'p': 0.6, 'q': 0.1, 'r': 0.8, 's': 0.1, 't': 0.4},  # V6
        ]

        return lead_factors[lead_idx] if lead_idx < len(lead_factors) else lead_factors[0]

    def _apply_lead_characteristics(self, ecg_data: np.ndarray) -> np.ndarray:
        """Apply realistic lead characteristics and scaling."""
        # Convert to millivolts (typical ECG amplitude range)
        ecg_data = ecg_data * 1.0  # Base amplitude in mV

        # Add slight lead-to-lead variation
        for lead_idx in range(ecg_data.shape[0]):
            variation = np.random.normal(1.0, 0.05)  # ±5% variation
            ecg_data[lead_idx] *= variation

        return ecg_data

    def generate_arrhythmia_example(self, arrhythmia_type: str = 'atrial_fibrillation') -> Tuple[np.ndarray, Dict]:
        """Generate example arrhythmias for testing."""
        if arrhythmia_type == 'atrial_fibrillation':
            return self._generate_atrial_fibrillation()
        elif arrhythmia_type == 'ventricular_tachycardia':
            return self._generate_ventricular_tachycardia()
        else:
            return self.generate_normal_sinus_rhythm()

    def _generate_atrial_fibrillation(self) -> Tuple[np.ndarray, Dict]:
        """Generate atrial fibrillation pattern."""
        # Irregular R-R intervals, no clear P waves
        duration = 10.0
        irregular_rr = np.random.uniform(0.4, 1.2, 15)  # Irregular intervals

        # Implementation would go here - for now return normal rhythm
        return self.generate_normal_sinus_rhythm(heart_rate=90)  # Placeholder

    def _generate_ventricular_tachycardia(self) -> Tuple[np.ndarray, Dict]:
        """Generate ventricular tachycardia pattern."""
        # Wide QRS complexes, high rate
        return self.generate_normal_sinus_rhythm(heart_rate=180)  # Placeholder


# Example usage and test functions
def create_example_datasets():
    """Create example ECG datasets for testing."""
    generator = ECGGenerator(sample_rate=500)

    examples = {}

    # Normal sinus rhythm examples
    examples['normal_72bpm'], _ = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=72, noise_level=0.05)

    examples['normal_60bpm'], _ = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=60, noise_level=0.03)

    examples['normal_100bpm'], _ = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=100, noise_level=0.05)

    # Noisy signal example
    examples['noisy_signal'], _ = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=75, noise_level=0.15)

    return examples


if __name__ == "__main__":
    # Generate and plot example ECG
    generator = ECGGenerator()
    ecg_data, metadata = generator.generate_normal_sinus_rhythm()

    print(f"Generated ECG: {metadata}")
    print(f"Shape: {ecg_data.shape}")
    print(f"Amplitude range: {ecg_data.min():.3f} to {ecg_data.max():.3f} mV")

    # Simple plotting test
    plt.figure(figsize=(12, 8))
    for i in range(min(4, ecg_data.shape[0])):
        plt.subplot(2, 2, i + 1)
        time_axis = np.linspace(0, metadata['duration'], ecg_data.shape[1])
        plt.plot(time_axis, ecg_data[i])
        plt.title(f"Lead {metadata['leads'][i]}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True)

    plt.tight_layout()
    plt.show()