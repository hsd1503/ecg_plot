#!/usr/bin/env python
"""
Advanced ECG Analysis Module
Heart Rate Variability (HRV) and Arrhythmia Detection.
"""

import numpy as np
from scipy import signal as sig, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from signal_processing import ECGSignalProcessor
from ecg_detection import ECGWaveDetector, ECGWaveFeatures


class RhythmType(Enum):
    """ECG rhythm classifications."""
    NORMAL_SINUS = "Normal Sinus Rhythm"
    SINUS_BRADY = "Sinus Bradycardia"
    SINUS_TACHY = "Sinus Tachycardia"
    ATRIAL_FIB = "Atrial Fibrillation"
    ATRIAL_FLUTTER = "Atrial Flutter"
    VTACH = "Ventricular Tachycardia"
    VFIB = "Ventricular Fibrillation"
    IRREGULAR = "Irregular Rhythm"
    UNKNOWN = "Unknown Rhythm"


@dataclass
class HRVMetrics:
    """Heart Rate Variability metrics."""
    # Time domain
    mean_rr: float  # Mean RR interval (ms)
    sdnn: float  # Standard deviation of NN intervals
    rmssd: float  # Root mean square of successive differences
    pnn50: float  # Percentage of successive RR intervals > 50ms

    # Frequency domain (simplified)
    lf_power: float  # Low frequency power
    hf_power: float  # High frequency power
    lf_hf_ratio: float  # LF/HF ratio

    # Nonlinear
    sd1: float  # Poincaré plot SD1
    sd2: float  # Poincaré plot SD2

    # Classification
    hrv_category: str  # "Excellent", "Good", "Fair", "Poor"


class HRVAnalyzer:
    """Heart Rate Variability analysis."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def analyze_hrv(self, rr_intervals: np.ndarray) -> HRVMetrics:
        """Complete HRV analysis."""
        # Convert to milliseconds
        rr_ms = rr_intervals * 1000

        # Time domain metrics
        mean_rr = np.mean(rr_ms)
        sdnn = np.std(rr_ms, ddof=1)

        # RMSSD
        successive_diffs = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))

        # pNN50
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

        # Frequency domain (simplified - using welch method)
        lf_power, hf_power, lf_hf_ratio = self._frequency_domain_analysis(rr_ms)

        # Poincaré plot metrics
        sd1, sd2 = self._poincare_analysis(rr_ms)

        # Classification
        hrv_category = self._classify_hrv(sdnn)

        return HRVMetrics(
            mean_rr=mean_rr,
            sdnn=sdnn,
            rmssd=rmssd,
            pnn50=pnn50,
            lf_power=lf_power,
            hf_power=hf_power,
            lf_hf_ratio=lf_hf_ratio,
            sd1=sd1,
            sd2=sd2,
            hrv_category=hrv_category
        )

    def _frequency_domain_analysis(self, rr_ms: np.ndarray) -> Tuple[float, float, float]:
        """Simplified frequency domain analysis."""
        if len(rr_ms) < 10:
            return 0.0, 0.0, 0.0

        # Resample to evenly spaced (required for FFT)
        # Using simple interpolation
        time_axis = np.cumsum(rr_ms) / 1000  # Convert to seconds
        fs_resample = 4  # 4 Hz resampling rate

        t_regular = np.arange(0, time_axis[-1], 1 / fs_resample)
        rr_regular = np.interp(t_regular, time_axis[:-1], rr_ms[:-1])

        # Power spectral density
        if len(rr_regular) > 10:
            frequencies, psd = sig.welch(rr_regular, fs_resample, nperseg=min(256, len(rr_regular)))

            # LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
            lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
            hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)

            lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])
            hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])

            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        else:
            lf_power = hf_power = lf_hf_ratio = 0.0

        return lf_power, hf_power, lf_hf_ratio

    def _poincare_analysis(self, rr_ms: np.ndarray) -> Tuple[float, float]:
        """Poincaré plot analysis."""
        if len(rr_ms) < 2:
            return 0.0, 0.0

        # SD1: standard deviation perpendicular to identity line
        successive_diffs = np.diff(rr_ms)
        sd1 = np.std(successive_diffs, ddof=1) / np.sqrt(2)

        # SD2: standard deviation along identity line
        sd2 = np.sqrt(2 * np.std(rr_ms, ddof=1) ** 2 - sd1 ** 2)

        return sd1, sd2

    def _classify_hrv(self, sdnn: float) -> str:
        """Classify HRV based on SDNN."""
        if sdnn > 100:
            return "Excellent"
        elif sdnn > 50:
            return "Good"
        elif sdnn > 25:
            return "Fair"
        else:
            return "Poor"


class ArrhythmiaDetector:
    """Automated arrhythmia detection."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.processor = ECGSignalProcessor(sample_rate)
        self.detector = ECGWaveDetector(sample_rate)

    def detect_rhythm(self, ecg_signal: np.ndarray, r_peaks: np.ndarray) -> Tuple[RhythmType, Dict]:
        """Detect cardiac rhythm from ECG."""
        analysis = {
            'heart_rate': 0.0,
            'rhythm_regularity': 0.0,
            'confidence': 0.0,
            'findings': []
        }

        if len(r_peaks) < 3:
            return RhythmType.UNKNOWN, analysis

        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sample_rate
        avg_hr = 60 / np.mean(rr_intervals)
        analysis['heart_rate'] = avg_hr

        # Assess rhythm regularity
        rr_std = np.std(rr_intervals)
        rr_cv = rr_std / np.mean(rr_intervals)  # Coefficient of variation
        analysis['rhythm_regularity'] = 1.0 - min(rr_cv, 1.0)

        # Rhythm classification logic
        rhythm_type = RhythmType.UNKNOWN
        confidence = 0.5

        # Check for normal sinus rhythm
        if 60 <= avg_hr <= 100 and rr_cv < 0.15:
            rhythm_type = RhythmType.NORMAL_SINUS
            confidence = 0.9
            analysis['findings'].append("Regular rhythm with normal rate")

        # Check for bradycardia
        elif avg_hr < 60 and rr_cv < 0.2:
            rhythm_type = RhythmType.SINUS_BRADY
            confidence = 0.85
            analysis['findings'].append(f"Bradycardia: HR {avg_hr:.0f} bpm")

        # Check for tachycardia
        elif avg_hr > 100 and rr_cv < 0.2:
            # Distinguish sinus vs ventricular tachycardia
            if avg_hr < 150:
                rhythm_type = RhythmType.SINUS_TACHY
                confidence = 0.85
                analysis['findings'].append(f"Sinus tachycardia: HR {avg_hr:.0f} bpm")
            else:
                rhythm_type = RhythmType.VTACH
                confidence = 0.7
                analysis['findings'].append(f"Possible VTach: HR {avg_hr:.0f} bpm")

        # Check for atrial fibrillation (irregular rhythm)
        elif rr_cv > 0.3:
            rhythm_type = RhythmType.ATRIAL_FIB
            confidence = 0.75
            analysis['findings'].append("Irregular rhythm suggestive of AFib")

        # Irregular but not clearly AFib
        elif rr_cv > 0.2:
            rhythm_type = RhythmType.IRREGULAR
            confidence = 0.7
            analysis['findings'].append("Irregular rhythm")

        analysis['confidence'] = confidence

        return rhythm_type, analysis

    def detect_pvc(self, features_list: List[ECGWaveFeatures]) -> List[int]:
        """Detect Premature Ventricular Contractions."""
        pvc_indices = []

        if len(features_list) < 3:
            return pvc_indices

        # Calculate mean RR interval
        rr_intervals = []
        for i in range(1, len(features_list)):
            if features_list[i].rr_interval:
                rr_intervals.append(features_list[i].rr_interval)

        if not rr_intervals:
            return pvc_indices

        mean_rr = np.mean(rr_intervals)

        # Look for early beats with wide QRS
        for i, features in enumerate(features_list):
            is_pvc = False

            # Early beat (RR < 80% of mean)
            if features.rr_interval and features.rr_interval < 0.8 * mean_rr:
                # Wide QRS (> 120ms)
                if features.qrs_duration:
                    qrs_ms = (features.qrs_duration / self.sample_rate) * 1000
                    if qrs_ms > 120:
                        is_pvc = True

            if is_pvc:
                pvc_indices.append(i)

        return pvc_indices

    def detect_st_changes(self, ecg_signal: np.ndarray, features: ECGWaveFeatures) -> Dict:
        """Detect ST segment elevation/depression."""
        st_analysis = {
            'st_elevation': 0.0,  # in mV
            'st_depression': 0.0,
            'is_significant': False,
            'interpretation': "Normal ST segment"
        }

        if not features.qrs_complex or not features.t_wave:
            return st_analysis

        # ST segment is between QRS offset and T-wave onset
        qrs_offset = features.qrs_complex[2]
        t_onset = features.t_wave[0]

        if t_onset <= qrs_offset:
            return st_analysis

        # Measure ST segment (J-point + 60-80ms)
        j_point = qrs_offset
        st_point = min(j_point + int(0.08 * self.sample_rate), t_onset)

        # Baseline (PR segment or isoelectric line)
        if features.p_wave:
            baseline = ecg_signal[features.p_wave[0]]
        else:
            baseline = 0.0

        st_level = ecg_signal[st_point] - baseline

        # Classify
        if st_level > 0.1:  # > 1mm elevation
            st_analysis['st_elevation'] = st_level
            st_analysis['is_significant'] = st_level > 0.2
            st_analysis['interpretation'] = f"ST elevation: {st_level:.2f} mV"
        elif st_level < -0.1:  # > 1mm depression
            st_analysis['st_depression'] = abs(st_level)
            st_analysis['is_significant'] = abs(st_level) > 0.1
            st_analysis['interpretation'] = f"ST depression: {abs(st_level):.2f} mV"

        return st_analysis


if __name__ == "__main__":
    from examples.generate_ecg_data import ECGGenerator

    # Test analysis
    gen = ECGGenerator()
    ecg_data, metadata = gen.generate_normal_sinus_rhythm(duration=60, heart_rate=75)

    processor = ECGSignalProcessor(metadata['sample_rate'])
    lead_ii = ecg_data[1]

    # Detect R-peaks
    r_peaks = processor.detect_r_peaks(lead_ii)
    rr_intervals = np.diff(r_peaks) / metadata['sample_rate']

    # HRV Analysis
    hrv_analyzer = HRVAnalyzer(metadata['sample_rate'])
    hrv_metrics = hrv_analyzer.analyze_hrv(rr_intervals)

    print("HRV Analysis:")
    print(f"  Mean RR: {hrv_metrics.mean_rr:.1f} ms")
    print(f"  SDNN: {hrv_metrics.sdnn:.1f} ms")
    print(f"  RMSSD: {hrv_metrics.rmssd:.1f} ms")
    print(f"  Category: {hrv_metrics.hrv_category}")

    # Rhythm Detection
    arrhythmia_detector = ArrhythmiaDetector(metadata['sample_rate'])
    rhythm_type, analysis = arrhythmia_detector.detect_rhythm(lead_ii, r_peaks)

    print(f"\nRhythm Analysis:")
    print(f"  Rhythm: {rhythm_type.value}")
    print(f"  Heart Rate: {analysis['heart_rate']:.1f} bpm")
    print(f"  Regularity: {analysis['rhythm_regularity']:.2f}")
    print(f"  Confidence: {analysis['confidence']:.2f}")

    print("\nAdvanced analysis working!")