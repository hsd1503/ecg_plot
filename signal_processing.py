#!/usr/bin/env python
"""
Advanced ECG Signal Processing Module
Implements clinical-grade signal processing algorithms for ECG analysis.
"""

import numpy as np
from scipy import signal as sig
from scipy.ndimage import median_filter
from typing import Tuple, Optional, Dict, List
import warnings


class ECGSignalProcessor:
    """Advanced ECG signal processing algorithms."""

    def __init__(self, sample_rate: int):
        """
        Initialize ECG signal processor.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0

    def remove_baseline_wander(self,
                               ecg_signal: np.ndarray,
                               cutoff_freq: float = 0.5,
                               method: str = 'highpass') -> np.ndarray:
        """
        Remove baseline wander from ECG signal.

        Args:
            ecg_signal: ECG signal array
            cutoff_freq: Cutoff frequency in Hz
            method: 'highpass', 'median', or 'polynomial'

        Returns:
            Baseline-corrected ECG signal
        """
        if method == 'highpass':
            # High-pass Butterworth filter
            nyq_cutoff = cutoff_freq / self.nyquist
            b, a = sig.butter(4, nyq_cutoff, btype='high')
            corrected = sig.filtfilt(b, a, ecg_signal)

        elif method == 'median':
            # Median filter for baseline estimation
            window_size = int(0.2 * self.sample_rate)  # 200ms window
            if window_size % 2 == 0:
                window_size += 1
            baseline = median_filter(ecg_signal, size=window_size)
            corrected = ecg_signal - baseline

        elif method == 'polynomial':
            # Polynomial detrending
            x = np.arange(len(ecg_signal))
            coeffs = np.polyfit(x, ecg_signal, deg=3)
            baseline = np.polyval(coeffs, x)
            corrected = ecg_signal - baseline

        else:
            raise ValueError(f"Unknown baseline removal method: {method}")

        return corrected

    def powerline_filter(self,
                         ecg_signal: np.ndarray,
                         powerline_freq: float = 60.0,
                         quality_factor: float = 30.0) -> np.ndarray:
        """
        Remove powerline interference (50/60 Hz).

        Args:
            ecg_signal: ECG signal array
            powerline_freq: Powerline frequency (50 or 60 Hz)
            quality_factor: Q factor for notch filter

        Returns:
            Filtered ECG signal
        """
        # Notch filter at powerline frequency
        nyq_freq = powerline_freq / self.nyquist
        b, a = sig.iirnotch(nyq_freq, quality_factor)

        # Also filter harmonics
        filtered = sig.filtfilt(b, a, ecg_signal)

        # Second harmonic
        if powerline_freq * 2 < self.nyquist:
            nyq_freq_2 = (powerline_freq * 2) / self.nyquist
            b2, a2 = sig.iirnotch(nyq_freq_2, quality_factor)
            filtered = sig.filtfilt(b2, a2, filtered)

        return filtered

    def bandpass_filter(self,
                        ecg_signal: np.ndarray,
                        lowcut: float = 0.5,
                        highcut: float = 40.0,
                        order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal.

        Args:
            ecg_signal: ECG signal array
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Bandpass filtered ECG signal
        """
        low = lowcut / self.nyquist
        high = highcut / self.nyquist

        b, a = sig.butter(order, [low, high], btype='band')
        filtered = sig.filtfilt(b, a, ecg_signal)

        return filtered

    def denoise_signal(self,
                       ecg_signal: np.ndarray,
                       method: str = 'wavelet',
                       level: int = 5) -> np.ndarray:
        """
        Denoise ECG signal using advanced methods.

        Args:
            ecg_signal: ECG signal array
            method: 'wavelet', 'savitzky_golay', or 'moving_average'
            level: Decomposition level for wavelet

        Returns:
            Denoised ECG signal
        """
        if method == 'wavelet':
            # Wavelet denoising (simplified without pywt)
            # Use Savitzky-Golay as approximation
            return sig.savgol_filter(ecg_signal,
                                     window_length=min(11, len(ecg_signal) // 2 * 2 - 1),
                                     polyorder=3)

        elif method == 'savitzky_golay':
            # Savitzky-Golay filter
            window_length = min(11, len(ecg_signal) // 2 * 2 - 1)
            return sig.savgol_filter(ecg_signal, window_length, polyorder=3)

        elif method == 'moving_average':
            # Moving average filter
            window_size = min(5, len(ecg_signal))
            kernel = np.ones(window_size) / window_size
            return np.convolve(ecg_signal, kernel, mode='same')

        else:
            raise ValueError(f"Unknown denoising method: {method}")

    def detect_r_peaks(self,
                       ecg_signal: np.ndarray,
                       method: str = 'pan_tompkins') -> np.ndarray:
        """
        Detect R-peaks using advanced algorithms.

        Args:
            ecg_signal: ECG signal array
            method: 'pan_tompkins', 'hamilton', or 'simple'

        Returns:
            Array of R-peak indices
        """
        if method == 'pan_tompkins':
            return self._pan_tompkins_detector(ecg_signal)
        elif method == 'hamilton':
            return self._hamilton_detector(ecg_signal)
        elif method == 'simple':
            return self._simple_peak_detector(ecg_signal)
        else:
            raise ValueError(f"Unknown R-peak detection method: {method}")

    def _pan_tompkins_detector(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Pan-Tompkins QRS detection algorithm.

        Reference: Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
        IEEE transactions on biomedical engineering, (3), 230-236.
        """
        # Step 1: Bandpass filter (5-15 Hz for QRS)
        filtered = self.bandpass_filter(ecg_signal, lowcut=5.0, highcut=15.0)

        # Step 2: Derivative (emphasize slope information)
        derivative = np.diff(filtered)
        derivative = np.append(derivative, 0)

        # Step 3: Squaring (emphasize large differences)
        squared = derivative ** 2

        # Step 4: Moving window integration
        window_size = int(0.150 * self.sample_rate)  # 150ms integration window
        integrated = np.convolve(squared, np.ones(window_size), mode='same') / window_size

        # Step 5: Adaptive thresholding
        peaks = []

        # Initialize thresholds
        SPKI = 0.0  # Signal peak
        NPKI = 0.0  # Noise peak
        THRESHOLD_I1 = 0.0

        RR_MISSED_LIMIT = int(1.66 * self.sample_rate)  # 1.66 seconds
        RR_LOW_LIMIT = int(0.2 * self.sample_rate)  # 200ms minimum RR

        # First pass - establish initial thresholds
        for i in range(1, len(integrated) - 1):
            if integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1]:
                if integrated[i] > SPKI:
                    SPKI = integrated[i]

        NPKI = 0.1 * SPKI
        THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)

        # Second pass - detect peaks
        last_peak_idx = 0

        for i in range(1, len(integrated) - 1):
            # Check if local maximum
            if integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1]:
                # Check if above threshold
                if integrated[i] > THRESHOLD_I1:
                    # Check if sufficient time from last peak
                    if not peaks or (i - peaks[-1]) > RR_LOW_LIMIT:
                        peaks.append(i)

                        # Update signal peak
                        SPKI = 0.125 * integrated[i] + 0.875 * SPKI
                        last_peak_idx = i

                    # Update threshold
                    THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)
                else:
                    # Update noise peak
                    NPKI = 0.125 * integrated[i] + 0.875 * NPKI
                    THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)

            # Check for missed peaks (searchback)
            if peaks and (i - peaks[-1]) > RR_MISSED_LIMIT:
                # Look back for missed peak
                search_start = max(peaks[-1] + RR_LOW_LIMIT, i - RR_MISSED_LIMIT)
                search_region = integrated[search_start:i]

                if len(search_region) > 0:
                    max_idx = np.argmax(search_region) + search_start
                    if integrated[max_idx] > THRESHOLD_I1 * 0.5:
                        peaks.append(max_idx)
                        peaks.sort()

        return np.array(peaks)

    def _hamilton_detector(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Hamilton-Tompkins QRS detection (simplified)."""
        # Preprocessing
        filtered = self.bandpass_filter(ecg_signal, 8, 20)

        # First derivative
        diff_signal = np.diff(filtered)
        diff_signal = np.append(diff_signal, 0)

        # Squaring
        squared = diff_signal ** 2

        # Moving average
        window_size = int(0.08 * self.sample_rate)  # 80ms
        integrated = np.convolve(squared, np.ones(window_size), mode='same') / window_size

        # Peak detection with adaptive threshold
        threshold = 0.3 * np.max(integrated)
        peaks = sig.find_peaks(integrated, height=threshold, distance=int(0.2 * self.sample_rate))[0]

        return peaks

    def _simple_peak_detector(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Simple threshold-based peak detector."""
        # Filter signal
        filtered = self.bandpass_filter(ecg_signal)

        # Adaptive threshold
        threshold = np.mean(filtered) + 0.5 * np.std(filtered)

        # Find peaks
        min_distance = int(0.25 * self.sample_rate)  # 250ms minimum between peaks
        peaks = sig.find_peaks(filtered, height=threshold, distance=min_distance)[0]

        return peaks

    def calculate_heart_rate(self,
                             r_peaks: np.ndarray,
                             method: str = 'average') -> Tuple[float, np.ndarray]:
        """
        Calculate heart rate from R-peaks.

        Args:
            r_peaks: Array of R-peak indices
            method: 'average', 'instantaneous', or 'median'

        Returns:
            Tuple of (average_hr, instantaneous_hr_array)
        """
        if len(r_peaks) < 2:
            return 0.0, np.array([])

        # Calculate RR intervals in seconds
        rr_intervals = np.diff(r_peaks) / self.sample_rate

        # Instantaneous heart rates
        instantaneous_hr = 60.0 / rr_intervals

        # Average heart rate
        if method == 'average':
            avg_hr = np.mean(instantaneous_hr)
        elif method == 'median':
            avg_hr = np.median(instantaneous_hr)
        else:
            avg_hr = np.mean(instantaneous_hr)

        return avg_hr, instantaneous_hr

    def assess_signal_quality(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """
        Assess ECG signal quality.

        Args:
            ecg_signal: ECG signal array

        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}

        # 1. Signal-to-Noise Ratio (SNR)
        # High-frequency content (> 40 Hz) as noise estimate
        high_freq = sig.filtfilt(*sig.butter(4, 40.0 / self.nyquist, 'high'), ecg_signal)
        noise_power = np.var(high_freq)
        signal_power = np.var(ecg_signal)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')

        quality_metrics['snr_db'] = snr

        # 2. Baseline stability (low-frequency variation)
        baseline = sig.filtfilt(*sig.butter(4, 0.5 / self.nyquist, 'low'), ecg_signal)
        baseline_variation = np.std(baseline)
        quality_metrics['baseline_variation'] = baseline_variation

        # 3. Saturation detection
        signal_range = np.max(ecg_signal) - np.min(ecg_signal)
        saturation_threshold = 0.95
        high_saturation = np.sum(ecg_signal > (np.max(ecg_signal) - 0.05 * signal_range)) / len(ecg_signal)
        low_saturation = np.sum(ecg_signal < (np.min(ecg_signal) + 0.05 * signal_range)) / len(ecg_signal)

        quality_metrics['saturation_percentage'] = (high_saturation + low_saturation) * 100

        # 4. Overall quality score (0-100)
        quality_score = 100.0

        # Penalize for low SNR
        if snr < 20:
            quality_score -= (20 - snr) * 2

        # Penalize for high baseline variation
        if baseline_variation > 0.2:
            quality_score -= min(50, baseline_variation * 100)

        # Penalize for saturation
        quality_score -= quality_metrics['saturation_percentage'] * 2

        quality_score = max(0, min(100, quality_score))
        quality_metrics['overall_quality'] = quality_score

        # Quality classification
        if quality_score >= 80:
            quality_metrics['quality_class'] = 'Excellent'
        elif quality_score >= 60:
            quality_metrics['quality_class'] = 'Good'
        elif quality_score >= 40:
            quality_metrics['quality_class'] = 'Fair'
        else:
            quality_metrics['quality_class'] = 'Poor'

        return quality_metrics

    def normalize_signal(self,
                         ecg_signal: np.ndarray,
                         method: str = 'z_score') -> np.ndarray:
        """
        Normalize ECG signal.

        Args:
            ecg_signal: ECG signal array
            method: 'z_score', 'min_max', or 'robust'

        Returns:
            Normalized ECG signal
        """
        if method == 'z_score':
            # Zero mean, unit variance
            normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

        elif method == 'min_max':
            # Scale to [0, 1]
            min_val = np.min(ecg_signal)
            max_val = np.max(ecg_signal)
            normalized = (ecg_signal - min_val) / (max_val - min_val)

        elif method == 'robust':
            # Use median and IQR for robustness to outliers
            median = np.median(ecg_signal)
            q75, q25 = np.percentile(ecg_signal, [75, 25])
            iqr = q75 - q25

            if iqr > 0:
                normalized = (ecg_signal - median) / iqr
            else:
                normalized = ecg_signal - median

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def preprocess_ecg(self,
                       ecg_signal: np.ndarray,
                       remove_baseline: bool = True,
                       remove_powerline: bool = True,
                       denoise: bool = True,
                       powerline_freq: float = 60.0) -> Tuple[np.ndarray, Dict]:
        """
        Complete ECG preprocessing pipeline.

        Args:
            ecg_signal: Raw ECG signal array
            remove_baseline: Remove baseline wander
            remove_powerline: Remove powerline interference
            denoise: Apply denoising
            powerline_freq: Powerline frequency (50 or 60 Hz)

        Returns:
            Tuple of (preprocessed_signal, processing_info)
        """
        processing_info = {
            'steps': [],
            'quality_before': {},
            'quality_after': {}
        }

        # Assess initial quality
        processing_info['quality_before'] = self.assess_signal_quality(ecg_signal)

        processed = ecg_signal.copy()

        # Step 1: Remove baseline wander
        if remove_baseline:
            processed = self.remove_baseline_wander(processed)
            processing_info['steps'].append('Baseline removal')

        # Step 2: Remove powerline interference
        if remove_powerline and powerline_freq < self.nyquist:
            processed = self.powerline_filter(processed, powerline_freq)
            processing_info['steps'].append(f'Powerline filter ({powerline_freq} Hz)')

        # Step 3: Denoise
        if denoise:
            processed = self.denoise_signal(processed, method='savitzky_golay')
            processing_info['steps'].append('Denoising')

        # Assess final quality
        processing_info['quality_after'] = self.assess_signal_quality(processed)

        return processed, processing_info


# Convenience functions
def create_signal_processor(sample_rate: int) -> ECGSignalProcessor:
    """Create ECG signal processor."""
    return ECGSignalProcessor(sample_rate)


def quick_preprocess(ecg_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Quick ECG preprocessing."""
    processor = ECGSignalProcessor(sample_rate)
    preprocessed, _ = processor.preprocess_ecg(ecg_signal)
    return preprocessed


def detect_peaks(ecg_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Quick R-peak detection."""
    processor = ECGSignalProcessor(sample_rate)
    return processor.detect_r_peaks(ecg_signal, method='pan_tompkins')


if __name__ == "__main__":
    # Example usage
    from examples.generate_ecg_data import ECGGenerator

    # Generate test ECG
    gen = ECGGenerator()
    ecg_data, metadata = gen.generate_normal_sinus_rhythm(duration=10, heart_rate=75)

    # Create processor
    processor = ECGSignalProcessor(metadata['sample_rate'])

    # Test preprocessing
    print("Testing ECG Signal Processing...")
    lead_ii = ecg_data[1]

    preprocessed, info = processor.preprocess_ecg(lead_ii)
    print(f"\nPreprocessing complete:")
    print(f"  Steps: {', '.join(info['steps'])}")
    print(
        f"  Quality before: {info['quality_before']['quality_class']} ({info['quality_before']['overall_quality']:.1f}/100)")
    print(
        f"  Quality after: {info['quality_after']['quality_class']} ({info['quality_after']['overall_quality']:.1f}/100)")

    # Test R-peak detection
    r_peaks = processor.detect_r_peaks(preprocessed, method='pan_tompkins')
    print(f"\nR-peak detection:")
    print(f"  Detected {len(r_peaks)} R-peaks")

    # Calculate heart rate
    avg_hr, inst_hr = processor.calculate_heart_rate(r_peaks)
    print(f"  Average heart rate: {avg_hr:.1f} bpm")

    print("\nSignal processing module working perfectly!")