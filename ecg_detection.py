#!/usr/bin/env python
"""
ECG Wave Detection Module
P-wave, QRS complex, and T-wave detection with clinical measurements.
"""

import numpy as np
from scipy import signal as sig
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from signal_processing import ECGSignalProcessor


@dataclass
class ECGWaveFeatures:
    """ECG wave features for a single beat."""
    r_peak: int
    p_wave: Optional[Tuple[int, int, int]] = None  # (onset, peak, offset)
    qrs_complex: Optional[Tuple[int, int, int]] = None  # (onset, peak, offset)
    t_wave: Optional[Tuple[int, int, int]] = None  # (onset, peak, offset)

    # Intervals (in samples)
    pr_interval: Optional[int] = None
    qrs_duration: Optional[int] = None
    qt_interval: Optional[int] = None
    rr_interval: Optional[int] = None

    # Amplitudes (in mV)
    p_amplitude: Optional[float] = None
    r_amplitude: Optional[float] = None
    s_amplitude: Optional[float] = None
    t_amplitude: Optional[float] = None

    # Quality
    detection_confidence: float = 0.0


class ECGWaveDetector:
    """Advanced ECG wave detection and feature extraction."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.processor = ECGSignalProcessor(sample_rate)

    def detect_all_waves(self, ecg_signal: np.ndarray) -> List[ECGWaveFeatures]:
        """Detect all ECG waves (P, QRS, T) for entire signal."""
        # Preprocess
        preprocessed, _ = self.processor.preprocess_ecg(ecg_signal)

        # Detect R-peaks
        r_peaks = self.processor.detect_r_peaks(preprocessed)

        # Extract features for each beat
        features_list = []
        for i, r_peak in enumerate(r_peaks):
            features = self._extract_beat_features(preprocessed, r_peak, r_peaks, i)
            features_list.append(features)

        return features_list

    def _extract_beat_features(self, signal: np.ndarray, r_peak: int,
                               all_r_peaks: np.ndarray, beat_index: int) -> ECGWaveFeatures:
        """Extract features for single beat."""
        features = ECGWaveFeatures(r_peak=r_peak)

        # Define search windows
        search_before = int(0.3 * self.sample_rate)  # 300ms before R
        search_after = int(0.4 * self.sample_rate)  # 400ms after R

        # P-wave detection (60-200ms before R)
        p_start = max(0, r_peak - search_before)
        p_end = r_peak - int(0.04 * self.sample_rate)  # At least 40ms before R

        if p_end > p_start:
            features.p_wave = self._detect_p_wave(signal, p_start, p_end)
            if features.p_wave:
                features.p_amplitude = signal[features.p_wave[1]]

        # QRS complex detection
        qrs_start = max(0, r_peak - int(0.05 * self.sample_rate))
        qrs_end = min(len(signal), r_peak + int(0.08 * self.sample_rate))

        features.qrs_complex = self._detect_qrs(signal, qrs_start, qrs_end, r_peak)
        if features.qrs_complex:
            features.qrs_duration = features.qrs_complex[2] - features.qrs_complex[0]
            features.r_amplitude = signal[r_peak]

            # Find S-wave (minimum after R-peak)
            s_search = signal[r_peak:qrs_end]
            if len(s_search) > 0:
                s_idx = r_peak + np.argmin(s_search)
                features.s_amplitude = signal[s_idx]

        # T-wave detection (120-400ms after R)
        t_start = r_peak + int(0.12 * self.sample_rate)
        t_end = min(len(signal), r_peak + search_after)

        if t_end > t_start:
            features.t_wave = self._detect_t_wave(signal, t_start, t_end)
            if features.t_wave:
                features.t_amplitude = signal[features.t_wave[1]]

        # Calculate intervals
        if features.p_wave and features.qrs_complex:
            features.pr_interval = features.qrs_complex[0] - features.p_wave[0]

        if features.qrs_complex and features.t_wave:
            features.qt_interval = features.t_wave[2] - features.qrs_complex[0]

        # RR interval
        if beat_index > 0:
            features.rr_interval = r_peak - all_r_peaks[beat_index - 1]

        # Detection confidence
        features.detection_confidence = self._calculate_confidence(features)

        return features

    def _detect_p_wave(self, signal: np.ndarray, start: int, end: int) -> Optional[Tuple[int, int, int]]:
        """Detect P-wave in specified window."""
        if end <= start:
            return None

        segment = signal[start:end]

        # Find peak
        peak_idx = start + np.argmax(segment)

        # Find onset (zero-crossing or minimum before peak)
        onset_search = segment[:peak_idx - start]
        if len(onset_search) > 0:
            onset_idx = start + np.argmin(onset_search)
        else:
            onset_idx = start

        # Find offset (zero-crossing or minimum after peak)
        offset_search = segment[peak_idx - start:]
        if len(offset_search) > 0:
            offset_idx = peak_idx + np.argmin(offset_search)
        else:
            offset_idx = end - 1

        return (onset_idx, peak_idx, offset_idx)

    def _detect_qrs(self, signal: np.ndarray, start: int, end: int, r_peak: int) -> Optional[Tuple[int, int, int]]:
        """Detect QRS complex boundaries."""
        # Q-point: minimum before R-peak
        q_search = signal[start:r_peak]
        if len(q_search) > 0:
            q_idx = start + np.argmin(q_search)
        else:
            q_idx = start

        # S-point: minimum after R-peak
        s_search = signal[r_peak:end]
        if len(s_search) > 0:
            s_idx = r_peak + np.argmin(s_search)
        else:
            s_idx = end - 1

        # Refine onset/offset based on slope
        onset = self._find_qrs_onset(signal, q_idx, r_peak)
        offset = self._find_qrs_offset(signal, r_peak, s_idx)

        return (onset, r_peak, offset)

    def _detect_t_wave(self, signal: np.ndarray, start: int, end: int) -> Optional[Tuple[int, int, int]]:
        """Detect T-wave in specified window."""
        if end <= start:
            return None

        segment = signal[start:end]

        # T-wave is usually the largest peak in this window
        peak_idx = start + np.argmax(np.abs(segment))

        # Find boundaries
        onset_idx = start
        offset_idx = end - 1

        # Refine boundaries based on slope change
        if peak_idx > start + 5:
            derivative = np.diff(signal[start:peak_idx])
            if len(derivative) > 0:
                # Onset where derivative increases significantly
                threshold = 0.1 * np.max(np.abs(derivative))
                onset_candidates = np.where(np.abs(derivative) > threshold)[0]
                if len(onset_candidates) > 0:
                    onset_idx = start + onset_candidates[0]

        if peak_idx < end - 5:
            derivative = np.diff(signal[peak_idx:end])
            if len(derivative) > 0:
                # Offset where derivative approaches zero
                threshold = 0.1 * np.max(np.abs(derivative))
                offset_candidates = np.where(np.abs(derivative) < threshold)[0]
                if len(offset_candidates) > 0:
                    offset_idx = peak_idx + offset_candidates[-1]

        return (onset_idx, peak_idx, offset_idx)

    def _find_qrs_onset(self, signal: np.ndarray, q_point: int, r_peak: int) -> int:
        """Find QRS onset using slope threshold."""
        if r_peak <= q_point + 2:
            return q_point

        # Calculate derivative
        segment = signal[q_point:r_peak]
        derivative = np.diff(segment)

        if len(derivative) == 0:
            return q_point

        # Find where slope exceeds threshold
        threshold = 0.15 * np.max(np.abs(derivative))
        candidates = np.where(np.abs(derivative) > threshold)[0]

        if len(candidates) > 0:
            return q_point + candidates[0]
        else:
            return q_point

    def _find_qrs_offset(self, signal: np.ndarray, r_peak: int, s_point: int) -> int:
        """Find QRS offset using slope threshold."""
        if s_point <= r_peak + 2:
            return s_point

        segment = signal[r_peak:s_point]
        derivative = np.diff(segment)

        if len(derivative) == 0:
            return s_point

        # Find where slope approaches zero
        threshold = 0.15 * np.max(np.abs(derivative))
        candidates = np.where(np.abs(derivative) < threshold)[0]

        if len(candidates) > 0:
            return r_peak + candidates[-1]
        else:
            return s_point

    def _calculate_confidence(self, features: ECGWaveFeatures) -> float:
        """Calculate detection confidence score (0-1)."""
        confidence = 1.0

        # Penalize missing waves
        if features.p_wave is None:
            confidence *= 0.9
        if features.qrs_complex is None:
            confidence *= 0.5  # QRS is critical
        if features.t_wave is None:
            confidence *= 0.9

        # Check physiological constraints
        if features.pr_interval:
            pr_ms = (features.pr_interval / self.sample_rate) * 1000
            if not (120 <= pr_ms <= 200):
                confidence *= 0.8

        if features.qrs_duration:
            qrs_ms = (features.qrs_duration / self.sample_rate) * 1000
            if not (80 <= qrs_ms <= 120):
                confidence *= 0.8

        return confidence

    def calculate_intervals_ms(self, features: ECGWaveFeatures) -> Dict[str, Optional[float]]:
        """Convert intervals to milliseconds."""
        intervals = {
            'PR': None,
            'QRS': None,
            'QT': None,
            'RR': None,
            'QTc': None  # Corrected QT (Bazett's formula)
        }

        if features.pr_interval:
            intervals['PR'] = (features.pr_interval / self.sample_rate) * 1000

        if features.qrs_duration:
            intervals['QRS'] = (features.qrs_duration / self.sample_rate) * 1000

        if features.qt_interval:
            intervals['QT'] = (features.qt_interval / self.sample_rate) * 1000

            # Calculate QTc if RR available
            if features.rr_interval:
                rr_s = features.rr_interval / self.sample_rate
                qt_s = features.qt_interval / self.sample_rate
                qtc_s = qt_s / np.sqrt(rr_s)  # Bazett's formula
                intervals['QTc'] = qtc_s * 1000

        if features.rr_interval:
            intervals['RR'] = (features.rr_interval / self.sample_rate) * 1000

        return intervals


if __name__ == "__main__":
    from examples.generate_ecg_data import ECGGenerator

    # Test wave detection
    gen = ECGGenerator()
    ecg_data, metadata = gen.generate_normal_sinus_rhythm(duration=10)

    detector = ECGWaveDetector(metadata['sample_rate'])
    features_list = detector.detect_all_waves(ecg_data[1])

    print(f"Detected features for {len(features_list)} beats")

    if features_list:
        first_beat = features_list[0]
        intervals = detector.calculate_intervals_ms(first_beat)

        print(f"\nFirst beat analysis:")
        print(f"  PR interval: {intervals['PR']:.1f} ms" if intervals['PR'] else "  PR: Not detected")
        print(f"  QRS duration: {intervals['QRS']:.1f} ms" if intervals['QRS'] else "  QRS: Not detected")
        print(f"  QT interval: {intervals['QT']:.1f} ms" if intervals['QT'] else "  QT: Not detected")
        print(f"  Confidence: {first_beat.detection_confidence:.2f}")

    print("\nWave detection working!")