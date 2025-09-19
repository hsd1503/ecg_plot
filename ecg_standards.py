#!/usr/bin/env python
"""
ECG Clinical Standards Module
Implements international ECG display standards for clinical practice.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ECGPaperStandard(Enum):
    """Standard ECG paper specifications."""
    STANDARD_25 = "25mm/s"  # Standard paper speed
    STANDARD_50 = "50mm/s"  # High-speed paper
    STANDARD_12_5 = "12.5mm/s"  # Slow paper speed


class ECGAmplitudeStandard(Enum):
    """Standard ECG amplitude scales."""
    STANDARD_10 = "10mm/mV"  # Standard amplitude scale
    HALF_GAIN_5 = "5mm/mV"  # Half gain
    DOUBLE_GAIN_20 = "20mm/mV"  # Double gain


@dataclass
class ECGDisplayParameters:
    """Clinical ECG display parameters."""
    paper_speed: float = 25.0  # mm/s
    amplitude_scale: float = 10.0  # mm/mV
    major_grid_time: float = 0.2  # seconds (5mm at 25mm/s)
    major_grid_voltage: float = 0.5  # mV (5mm at 10mm/mV)
    minor_grid_time: float = 0.04  # seconds (1mm at 25mm/s)
    minor_grid_voltage: float = 0.1  # mV (1mm at 10mm/mV)

    def __post_init__(self):
        """Calculate derived parameters."""
        # Grid spacing calculations
        self.major_grid_time = 0.2 * (25.0 / self.paper_speed)
        self.minor_grid_time = 0.04 * (25.0 / self.paper_speed)
        self.major_grid_voltage = 0.5 * (10.0 / self.amplitude_scale)
        self.minor_grid_voltage = 0.1 * (10.0 / self.amplitude_scale)


class ClinicalECGStandards:
    """Implements clinical ECG standards and conversions."""

    # Standard lead configurations
    STANDARD_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Clinical display layouts
    CLINICAL_LAYOUTS = {
        '4x3_standard': {
            'description': 'Standard 4-row, 3-column clinical layout',
            'layout': [
                ['I', 'aVR', 'V1', 'V4'],
                ['II', 'aVL', 'V2', 'V5'],
                ['III', 'aVF', 'V3', 'V6'],
                ['II', 'II', 'II', 'II']  # Rhythm strip (Lead II)
            ]
        },
        '3x4_landscape': {
            'description': '3-row, 4-column landscape layout',
            'layout': [
                ['I', 'II', 'III', 'aVR'],
                ['aVL', 'aVF', 'V1', 'V2'],
                ['V3', 'V4', 'V5', 'V6']
            ]
        },
        '6x2_compact': {
            'description': '6-row, 2-column compact layout',
            'layout': [
                ['I', 'V1'],
                ['II', 'V2'],
                ['III', 'V3'],
                ['aVR', 'V4'],
                ['aVL', 'V5'],
                ['aVF', 'V6']
            ]
        }
    }

    def __init__(self):
        """Initialize clinical standards."""
        self.display_params = ECGDisplayParameters()

    def convert_to_clinical_scale(self,
                                  ecg_data: np.ndarray,
                                  original_sample_rate: int,
                                  target_paper_speed: float = 25.0,
                                  target_amplitude_scale: float = 10.0) -> Tuple[np.ndarray, Dict]:
        """
        Convert ECG data to clinical paper standards.

        Args:
            ecg_data: ECG signal array (leads, samples)
            original_sample_rate: Original sampling rate in Hz
            target_paper_speed: Target paper speed in mm/s
            target_amplitude_scale: Target amplitude scale in mm/mV

        Returns:
            Tuple of (scaled_ecg_data, conversion_info)
        """
        # Calculate time scaling factor
        # Standard: 25mm/s means 1 second = 25mm on paper
        time_scale_factor = target_paper_speed / 25.0

        # Calculate amplitude scaling
        # Standard: 10mm/mV means 1mV = 10mm on paper
        amplitude_scale_factor = target_amplitude_scale / 10.0

        # Apply scaling
        scaled_ecg = ecg_data * amplitude_scale_factor

        conversion_info = {
            'original_sample_rate': original_sample_rate,
            'time_scale_factor': time_scale_factor,
            'amplitude_scale_factor': amplitude_scale_factor,
            'paper_speed_mm_per_s': target_paper_speed,
            'amplitude_scale_mm_per_mv': target_amplitude_scale,
            'grid_major_time_s': 0.2 * time_scale_factor,
            'grid_major_voltage_mv': 0.5 / amplitude_scale_factor,
            'grid_minor_time_s': 0.04 * time_scale_factor,
            'grid_minor_voltage_mv': 0.1 / amplitude_scale_factor
        }

        return scaled_ecg, conversion_info

    def get_clinical_grid_parameters(self,
                                     duration: float,
                                     amplitude_range: Tuple[float, float],
                                     paper_speed: float = 25.0,
                                     amplitude_scale: float = 10.0) -> Dict:
        """
        Calculate clinical grid parameters.

        Args:
            duration: Signal duration in seconds
            amplitude_range: (min_mv, max_mv) amplitude range
            paper_speed: Paper speed in mm/s
            amplitude_scale: Amplitude scale in mm/mV

        Returns:
            Dictionary with grid parameters
        """
        min_mv, max_mv = amplitude_range

        # Time grid parameters
        major_time_interval = 0.2 * (25.0 / paper_speed)  # 5mm intervals
        minor_time_interval = 0.04 * (25.0 / paper_speed)  # 1mm intervals

        # Voltage grid parameters
        major_voltage_interval = 0.5 * (10.0 / amplitude_scale)  # 5mm intervals
        minor_voltage_interval = 0.1 * (10.0 / amplitude_scale)  # 1mm intervals

        # Calculate grid ranges
        time_max = np.ceil(duration / major_time_interval) * major_time_interval
        voltage_max = np.ceil(max_mv / major_voltage_interval) * major_voltage_interval
        voltage_min = np.floor(min_mv / major_voltage_interval) * major_voltage_interval

        return {
            'time_range': (0, time_max),
            'voltage_range': (voltage_min, voltage_max),
            'major_time_ticks': np.arange(0, time_max + major_time_interval / 2, major_time_interval),
            'minor_time_ticks': np.arange(0, time_max + minor_time_interval / 2, minor_time_interval),
            'major_voltage_ticks': np.arange(voltage_min, voltage_max + major_voltage_interval / 2,
                                             major_voltage_interval),
            'minor_voltage_ticks': np.arange(voltage_min, voltage_max + minor_voltage_interval / 2,
                                             minor_voltage_interval),
            'paper_speed_mm_per_s': paper_speed,
            'amplitude_scale_mm_per_mv': amplitude_scale
        }

    def get_clinical_lead_layout(self,
                                 layout_type: str = '4x3_standard',
                                 rhythm_strip_lead: str = 'II') -> Dict:
        """
        Get clinical lead arrangement.

        Args:
            layout_type: Type of clinical layout
            rhythm_strip_lead: Lead to use for rhythm strip

        Returns:
            Dictionary with layout information
        """
        if layout_type not in self.CLINICAL_LAYOUTS:
            raise ValueError(f"Unknown layout type: {layout_type}. "
                             f"Available: {list(self.CLINICAL_LAYOUTS.keys())}")

        layout_info = self.CLINICAL_LAYOUTS[layout_type].copy()

        # Replace rhythm strip placeholder with specified lead
        for row_idx, row in enumerate(layout_info['layout']):
            for col_idx, lead in enumerate(row):
                if lead == 'II' and layout_type == '4x3_standard' and row_idx == 3:
                    layout_info['layout'][row_idx] = [rhythm_strip_lead] * 4

        # Add layout metadata
        layout_info.update({
            'rows': len(layout_info['layout']),
            'cols': len(layout_info['layout'][0]) if layout_info['layout'] else 0,
            'rhythm_strip_lead': rhythm_strip_lead,
            'total_plots': sum(len(row) for row in layout_info['layout']),
            'unique_leads': list(set([lead for row in layout_info['layout'] for lead in row]))
        })

        return layout_info

    def create_calibration_signal(self,
                                  sample_rate: int,
                                  amplitude_mv: float = 1.0,
                                  duration_s: float = 0.2,
                                  placement: str = 'beginning') -> Tuple[np.ndarray, Dict]:
        """
        Create standard ECG calibration signal.

        Args:
            sample_rate: Sampling rate in Hz
            amplitude_mv: Calibration amplitude in mV (typically 1.0)
            duration_s: Duration of calibration pulse in seconds
            placement: Where to place calibration ('beginning', 'end', 'both')

        Returns:
            Tuple of (calibration_signal, calibration_info)
        """
        n_samples = int(duration_s * sample_rate)

        # Create square wave calibration pulse
        calibration = np.zeros(n_samples)
        pulse_samples = int(0.1 * sample_rate)  # 100ms pulse width

        if placement in ['beginning', 'both']:
            calibration[:pulse_samples] = amplitude_mv

        if placement in ['end', 'both']:
            calibration[-pulse_samples:] = amplitude_mv

        calibration_info = {
            'amplitude_mv': amplitude_mv,
            'duration_s': duration_s,
            'pulse_width_s': 0.1,
            'sample_rate': sample_rate,
            'placement': placement,
            'n_samples': n_samples
        }

        return calibration, calibration_info

    def validate_clinical_compliance(self,
                                     ecg_data: np.ndarray,
                                     sample_rate: int,
                                     paper_speed: float = 25.0,
                                     amplitude_scale: float = 10.0) -> Dict[str, bool]:
        """
        Validate ECG data compliance with clinical standards.

        Args:
            ecg_data: ECG signal array
            sample_rate: Sampling rate
            paper_speed: Target paper speed
            amplitude_scale: Target amplitude scale

        Returns:
            Dictionary with compliance results
        """
        compliance = {
            'paper_speed_standard': False,
            'amplitude_scale_standard': False,
            'grid_compliance': False,
            'lead_count_standard': False,
            'signal_duration_adequate': False,
            'overall_compliant': False
        }

        # Check paper speed (should allow for reasonable sample rates)
        min_required_sr = paper_speed * 10  # At least 10 samples per mm
        max_reasonable_sr = paper_speed * 100  # Not more than 100 samples per mm
        compliance['paper_speed_standard'] = min_required_sr <= sample_rate <= max_reasonable_sr

        # Check amplitude scale (data should be in reasonable mV range)
        data_range = np.max(ecg_data) - np.min(ecg_data)
        compliance['amplitude_scale_standard'] = 0.5 <= data_range <= 10.0  # Reasonable ECG range

        # Check grid compliance (can we create appropriate grid?)
        duration = ecg_data.shape[1] / sample_rate
        amplitude_range = (np.min(ecg_data), np.max(ecg_data))
        try:
            grid_params = self.get_clinical_grid_parameters(duration, amplitude_range, paper_speed, amplitude_scale)
            compliance['grid_compliance'] = True
        except:
            compliance['grid_compliance'] = False

        # Check lead count
        n_leads = ecg_data.shape[0]
        compliance['lead_count_standard'] = n_leads in [1, 3, 6, 12, 15]  # Standard lead counts

        # Check signal duration (at least 2.5 seconds for meaningful ECG)
        compliance['signal_duration_adequate'] = duration >= 2.5

        # Overall compliance
        compliance['overall_compliant'] = all([
            compliance['paper_speed_standard'],
            compliance['amplitude_scale_standard'],
            compliance['grid_compliance'],
            compliance['lead_count_standard'],
            compliance['signal_duration_adequate']
        ])

        return compliance

    def get_measurement_scales(self,
                               sample_rate: int,
                               paper_speed: float = 25.0,
                               amplitude_scale: float = 10.0) -> Dict[str, float]:
        """
        Get measurement conversion scales.

        Args:
            sample_rate: Sampling rate in Hz
            paper_speed: Paper speed in mm/s
            amplitude_scale: Amplitude scale in mm/mV

        Returns:
            Dictionary with conversion factors
        """
        # Time conversions
        samples_per_second = sample_rate
        mm_per_second = paper_speed
        samples_per_mm = samples_per_second / mm_per_second

        # Amplitude conversions
        mm_per_mv = amplitude_scale

        return {
            'samples_per_mm_time': samples_per_mm,
            'mm_per_sample_time': 1.0 / samples_per_mm,
            'ms_per_mm': 1000.0 / paper_speed,
            'mm_per_ms': paper_speed / 1000.0,
            'mm_per_mv': mm_per_mv,
            'mv_per_mm': 1.0 / mm_per_mv,
            'samples_per_ms': sample_rate / 1000.0,
            'ms_per_sample': 1000.0 / sample_rate
        }


# Convenience functions for quick access
def get_standard_display_parameters() -> ECGDisplayParameters:
    """Get standard ECG display parameters (25mm/s, 10mm/mV)."""
    return ECGDisplayParameters()


def convert_ecg_to_standard(ecg_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict]:
    """Quick conversion to standard clinical scale."""
    standards = ClinicalECGStandards()
    return standards.convert_to_clinical_scale(ecg_data, sample_rate)


def get_clinical_grid(duration: float, amplitude_range: Tuple[float, float]) -> Dict:
    """Quick clinical grid parameters."""
    standards = ClinicalECGStandards()
    return standards.get_clinical_grid_parameters(duration, amplitude_range)


def get_standard_12lead_layout() -> Dict:
    """Get standard 4x3 clinical layout."""
    standards = ClinicalECGStandards()
    return standards.get_clinical_lead_layout('4x3_standard')


if __name__ == "__main__":
    # Example usage and testing
    standards = ClinicalECGStandards()

    # Test clinical grid parameters
    grid_params = standards.get_clinical_grid_parameters(
        duration=10.0,
        amplitude_range=(-2.0, 3.0)
    )
    print("Clinical Grid Parameters:")
    for key, value in grid_params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {len(value)} values from {value[0]:.3f} to {value[-1]:.3f}")
        else:
            print(f"  {key}: {value}")

    # Test lead layout
    layout = standards.get_clinical_lead_layout('4x3_standard')
    print(f"\nClinical Layout ({layout['description']}):")
    for i, row in enumerate(layout['layout']):
        print(f"  Row {i + 1}: {row}")

    # Test calibration signal
    cal_signal, cal_info = standards.create_calibration_signal(500, amplitude_mv=1.0)
    print(f"\nCalibration Signal:")
    print(f"  Duration: {cal_info['duration_s']}s")
    print(f"  Amplitude: {cal_info['amplitude_mv']}mV")
    print(f"  Samples: {cal_info['n_samples']}")