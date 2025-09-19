#!/usr/bin/env python
"""
ECG Clinical Measurements Module
Professional ECG measurement tools for clinical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from ecg_standards import ClinicalECGStandards


class MeasurementType(Enum):
    """Types of ECG measurements."""
    PR_INTERVAL = "PR Interval"
    QRS_DURATION = "QRS Duration"
    QT_INTERVAL = "QT Interval"
    QTC_INTERVAL = "QTc Interval"
    RR_INTERVAL = "RR Interval"
    HEART_RATE = "Heart Rate"
    CUSTOM_INTERVAL = "Custom Interval"


@dataclass
class ECGMeasurement:
    """ECG measurement result."""
    measurement_type: MeasurementType
    value: float
    unit: str
    normal_range: Tuple[float, float]
    is_normal: bool
    confidence: float
    start_sample: int
    end_sample: int
    lead_name: str
    notes: Optional[str] = None

    def __str__(self):
        status = "NORMAL" if self.is_normal else "ABNORMAL"
        return f"{self.measurement_type.value}: {self.value:.1f} {self.unit} ({status})"


class ECGMeasurementTools:
    """Professional ECG measurement and analysis tools."""

    def __init__(self, sample_rate: int, paper_speed: float = 25.0, amplitude_scale: float = 10.0):
        """
        Initialize ECG measurement tools.

        Args:
            sample_rate: Sampling rate in Hz
            paper_speed: Paper speed in mm/s
            amplitude_scale: Amplitude scale in mm/mV
        """
        self.sample_rate = sample_rate
        self.paper_speed = paper_speed
        self.amplitude_scale = amplitude_scale
        self.standards = ClinicalECGStandards()

        # Get conversion scales
        self.scales = self.standards.get_measurement_scales(
            sample_rate, paper_speed, amplitude_scale
        )

        # Normal ranges for clinical measurements (adult values)
        self.normal_ranges = {
            MeasurementType.PR_INTERVAL: (120, 200),  # ms
            MeasurementType.QRS_DURATION: (80, 120),  # ms
            MeasurementType.QT_INTERVAL: (350, 450),  # ms (rate dependent)
            MeasurementType.QTC_INTERVAL: (350, 450),  # ms (corrected)
            MeasurementType.RR_INTERVAL: (600, 1200),  # ms (60-100 bpm)
            MeasurementType.HEART_RATE: (60, 100),  # bpm
        }

    def add_measurement_calipers(self,
                                 ax: plt.Axes,
                                 start_time: float,
                                 end_time: float,
                                 measurement_type: MeasurementType = MeasurementType.CUSTOM_INTERVAL,
                                 y_position: Optional[float] = None,
                                 color: str = 'red',
                                 label: Optional[str] = None) -> Dict:
        """
        Add measurement calipers to ECG plot.

        Args:
            ax: Matplotlib axes object
            start_time: Start time in seconds
            end_time: End time in seconds
            measurement_type: Type of measurement
            y_position: Y position for calipers (auto if None)
            color: Caliper color
            label: Custom label

        Returns:
            Dictionary with measurement information
        """
        # Calculate measurement
        duration_ms = (end_time - start_time) * 1000
        duration_mm = (end_time - start_time) * self.paper_speed

        # Auto-position if not specified
        if y_position is None:
            ylim = ax.get_ylim()
            y_position = ylim[1] - 0.1 * (ylim[1] - ylim[0])

        # Draw calipers
        caliper_height = 0.2  # Height of caliper marks in mV

        # Vertical lines
        ax.axvline(start_time, color=color, linewidth=1.5, alpha=0.8)
        ax.axvline(end_time, color=color, linewidth=1.5, alpha=0.8)

        # Horizontal measurement line
        ax.plot([start_time, end_time], [y_position, y_position],
                color=color, linewidth=2, alpha=0.8)

        # Caliper marks (small vertical lines)
        ax.plot([start_time, start_time],
                [y_position - caliper_height / 2, y_position + caliper_height / 2],
                color=color, linewidth=2)
        ax.plot([end_time, end_time],
                [y_position - caliper_height / 2, y_position + caliper_height / 2],
                color=color, linewidth=2)

        # Label
        mid_time = (start_time + end_time) / 2
        if label is None:
            if measurement_type == MeasurementType.CUSTOM_INTERVAL:
                label = f"{duration_ms:.0f} ms"
            else:
                label = f"{measurement_type.value}: {duration_ms:.0f} ms"

        ax.text(mid_time, y_position + caliper_height, label,
                ha='center', va='bottom', fontsize=8,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.8))

        # Create measurement result
        measurement_info = {
            'start_time': start_time,
            'end_time': end_time,
            'duration_ms': duration_ms,
            'duration_mm': duration_mm,
            'measurement_type': measurement_type,
            'y_position': y_position,
            'label': label
        }

        return measurement_info

    def measure_rr_intervals(self,
                             ecg_signal: np.ndarray,
                             lead_name: str = "II",
                             method: str = 'simple_threshold') -> List[ECGMeasurement]:
        """
        Measure RR intervals and calculate heart rate.

        Args:
            ecg_signal: ECG signal array
            lead_name: Name of the lead
            method: Peak detection method

        Returns:
            List of ECGMeasurement objects
        """
        measurements = []

        try:
            # Simple R-peak detection
            r_peaks = self._detect_r_peaks(ecg_signal, method)

            if len(r_peaks) < 2:
                warnings.warn("Insufficient R peaks detected for RR measurement")
                return measurements

            # Calculate RR intervals
            rr_samples = np.diff(r_peaks)
            rr_intervals_ms = (rr_samples / self.sample_rate) * 1000

            # Calculate heart rates
            heart_rates = 60000 / rr_intervals_ms  # Convert to bpm

            # Create measurements for each RR interval
            for i, (rr_ms, hr) in enumerate(zip(rr_intervals_ms, heart_rates)):
                # RR interval measurement
                rr_normal_range = self.normal_ranges[MeasurementType.RR_INTERVAL]
                rr_is_normal = rr_normal_range[0] <= rr_ms <= rr_normal_range[1]

                rr_measurement = ECGMeasurement(
                    measurement_type=MeasurementType.RR_INTERVAL,
                    value=rr_ms,
                    unit="ms",
                    normal_range=rr_normal_range,
                    is_normal=rr_is_normal,
                    confidence=0.8,  # Simple confidence score
                    start_sample=r_peaks[i],
                    end_sample=r_peaks[i + 1],
                    lead_name=lead_name,
                    notes=f"RR interval #{i + 1}"
                )
                measurements.append(rr_measurement)

                # Heart rate measurement
                hr_normal_range = self.normal_ranges[MeasurementType.HEART_RATE]
                hr_is_normal = hr_normal_range[0] <= hr <= hr_normal_range[1]

                hr_measurement = ECGMeasurement(
                    measurement_type=MeasurementType.HEART_RATE,
                    value=hr,
                    unit="bpm",
                    normal_range=hr_normal_range,
                    is_normal=hr_is_normal,
                    confidence=0.8,
                    start_sample=r_peaks[i],
                    end_sample=r_peaks[i + 1],
                    lead_name=lead_name,
                    notes=f"HR from RR #{i + 1}"
                )
                measurements.append(hr_measurement)

        except Exception as e:
            warnings.warn(f"RR interval measurement failed: {e}")

        return measurements

    def measure_custom_interval(self,
                                start_sample: int,
                                end_sample: int,
                                measurement_type: MeasurementType = MeasurementType.CUSTOM_INTERVAL,
                                lead_name: str = "Unknown") -> ECGMeasurement:
        """
        Measure custom interval between two points.

        Args:
            start_sample: Start sample index
            end_sample: End sample index
            measurement_type: Type of measurement
            lead_name: Lead name

        Returns:
            ECGMeasurement object
        """
        # Calculate duration
        duration_samples = end_sample - start_sample
        duration_ms = (duration_samples / self.sample_rate) * 1000

        # Get normal range
        normal_range = self.normal_ranges.get(measurement_type, (0, float('inf')))
        is_normal = normal_range[0] <= duration_ms <= normal_range[1]

        measurement = ECGMeasurement(
            measurement_type=measurement_type,
            value=duration_ms,
            unit="ms",
            normal_range=normal_range,
            is_normal=is_normal,
            confidence=1.0,  # Manual measurement has high confidence
            start_sample=start_sample,
            end_sample=end_sample,
            lead_name=lead_name,
            notes="Manual measurement"
        )

        return measurement

    def add_measurement_grid(self,
                             ax: plt.Axes,
                             show_time_markers: bool = True,
                             show_amplitude_markers: bool = True,
                             marker_interval_ms: int = 200) -> Dict:
        """
        Add measurement grid with time and amplitude markers.

        Args:
            ax: Matplotlib axes object
            show_time_markers: Show time measurement markers
            show_amplitude_markers: Show amplitude markers
            marker_interval_ms: Interval for time markers in ms

        Returns:
            Dictionary with grid information
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        grid_info = {
            'time_markers': [],
            'amplitude_markers': [],
            'scales': self.scales.copy()
        }

        if show_time_markers:
            # Add time markers at specified intervals
            marker_interval_s = marker_interval_ms / 1000.0
            time_markers = np.arange(0, xlim[1], marker_interval_s)

            for t in time_markers:
                if xlim[0] <= t <= xlim[1]:
                    ax.axvline(t, color='blue', alpha=0.3, linewidth=0.5, linestyle='--')

                    # Add time label
                    ax.text(t, ylim[1] - 0.05 * (ylim[1] - ylim[0]),
                            f'{t * 1000:.0f}ms',
                            rotation=90, ha='center', va='top',
                            fontsize=6, alpha=0.7, color='blue')

                    grid_info['time_markers'].append(t)

        if show_amplitude_markers:
            # Add amplitude markers at 0.5mV intervals
            amp_interval = 0.5  # mV
            amp_markers = np.arange(
                np.ceil(ylim[0] / amp_interval) * amp_interval,
                np.floor(ylim[1] / amp_interval) * amp_interval + amp_interval,
                amp_interval
            )

            for amp in amp_markers:
                if ylim[0] <= amp <= ylim[1] and amp != 0:
                    ax.axhline(amp, color='green', alpha=0.3, linewidth=0.5, linestyle='--')

                    # Add amplitude label
                    ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), amp,
                            f'{amp:.1f}mV',
                            ha='left', va='center',
                            fontsize=6, alpha=0.7, color='green',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                    grid_info['amplitude_markers'].append(amp)

        return grid_info

    def create_measurements_report(self,
                                   measurements: List[ECGMeasurement],
                                   title: str = "ECG Measurements Report") -> str:
        """
        Create a formatted measurements report.

        Args:
            measurements: List of ECG measurements
            title: Report title

        Returns:
            Formatted report string
        """
        report = f"{title}\n"
        report += "=" * len(title) + "\n\n"

        # Group measurements by type
        by_type = {}
        for measurement in measurements:
            measurement_type = measurement.measurement_type
            if measurement_type not in by_type:
                by_type[measurement_type] = []
            by_type[measurement_type].append(measurement)

        # Generate report sections
        for measurement_type, type_measurements in by_type.items():
            report += f"{measurement_type.value}:\n"
            report += "-" * (len(measurement_type.value) + 1) + "\n"

            for i, measurement in enumerate(type_measurements):
                status_indicator = "✓" if measurement.is_normal else "⚠"
                report += f"  {status_indicator} {measurement}\n"

                if measurement.notes:
                    report += f"    Notes: {measurement.notes}\n"

                normal_range_str = f"{measurement.normal_range[0]}-{measurement.normal_range[1]} {measurement.unit}"
                report += f"    Normal range: {normal_range_str}\n"
                report += f"    Confidence: {measurement.confidence:.1%}\n"

                if i < len(type_measurements) - 1:
                    report += "\n"

            report += "\n"

        # Summary statistics
        normal_count = sum(1 for m in measurements if m.is_normal)
        abnormal_count = len(measurements) - normal_count

        report += "Summary:\n"
        report += "--------\n"
        report += f"Total measurements: {len(measurements)}\n"
        report += f"Normal: {normal_count}\n"
        report += f"Abnormal: {abnormal_count}\n"
        report += f"Overall assessment: {'NORMAL' if abnormal_count == 0 else 'ABNORMAL FINDINGS'}\n"

        return report

    def _detect_r_peaks(self, signal: np.ndarray, method: str = 'simple_threshold') -> np.ndarray:
        """
        Simple R-peak detection algorithm.

        Args:
            signal: ECG signal
            method: Detection method

        Returns:
            Array of R-peak sample indices
        """
        if method == 'simple_threshold':
            # Simple threshold-based peak detection
            signal_std = np.std(signal)
            threshold = np.mean(signal) + 2 * signal_std

            # Find peaks above threshold
            peaks = []
            min_distance = int(0.4 * self.sample_rate)  # Minimum 400ms between peaks

            for i in range(min_distance, len(signal) - min_distance):
                if (signal[i] > threshold and
                        signal[i] > signal[i - 1] and
                        signal[i] > signal[i + 1]):

                    # Check minimum distance from previous peak
                    if not peaks or i - peaks[-1] > min_distance:
                        peaks.append(i)

            return np.array(peaks)

        else:
            raise ValueError(f"Unknown peak detection method: {method}")

    def convert_samples_to_time(self, sample_indices: np.ndarray) -> np.ndarray:
        """Convert sample indices to time in seconds."""
        return sample_indices / self.sample_rate

    def convert_samples_to_ms(self, sample_indices: np.ndarray) -> np.ndarray:
        """Convert sample indices to time in milliseconds."""
        return (sample_indices / self.sample_rate) * 1000

    def convert_time_to_samples(self, time_seconds: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert time in seconds to sample indices."""
        samples = time_seconds * self.sample_rate
        if isinstance(samples, np.ndarray):
            return samples.astype(int)
        else:
            return int(samples)

    def convert_mm_to_time(self, mm: float) -> float:
        """Convert paper distance (mm) to time (seconds)."""
        return mm / self.paper_speed

    def convert_mm_to_mv(self, mm: float) -> float:
        """Convert paper distance (mm) to amplitude (mV)."""
        return mm / self.amplitude_scale


# Interactive measurement class for matplotlib widgets
class InteractiveMeasurementTool:
    """Interactive measurement tool with click-to-measure functionality."""

    def __init__(self, ax: plt.Axes, ecg_tools: ECGMeasurementTools):
        """Initialize interactive measurement tool."""
        self.ax = ax
        self.tools = ecg_tools
        self.start_point = None
        self.measurements = []

        # Connect mouse events
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        """Handle mouse click events for measurements."""
        if event.inaxes != self.ax:
            return

        if self.start_point is None:
            # First click - set start point
            self.start_point = (event.xdata, event.ydata)
            self.ax.axvline(event.xdata, color='red', alpha=0.7, linestyle='--')
            self.ax.figure.canvas.draw()
            print(f"Start point set at {event.xdata:.3f}s")

        else:
            # Second click - complete measurement
            end_time = event.xdata
            start_time = self.start_point[0]

            # Add calipers
            measurement_info = self.tools.add_measurement_calipers(
                self.ax, start_time, end_time, color='red'
            )

            self.measurements.append(measurement_info)
            self.ax.figure.canvas.draw()

            print(f"Measurement: {measurement_info['duration_ms']:.1f} ms")

            # Reset for next measurement
            self.start_point = None

    def get_measurements(self) -> List[Dict]:
        """Get all completed measurements."""
        return self.measurements.copy()


# Convenience functions
def create_measurement_tools(sample_rate: int,
                             paper_speed: float = 25.0,
                             amplitude_scale: float = 10.0) -> ECGMeasurementTools:
    """Create ECG measurement tools with specified parameters."""
    return ECGMeasurementTools(sample_rate, paper_speed, amplitude_scale)


def add_quick_calipers(ax: plt.Axes,
                       start_time: float,
                       end_time: float,
                       sample_rate: int,
                       label: Optional[str] = None) -> Dict:
    """Quick function to add measurement calipers."""
    tools = ECGMeasurementTools(sample_rate)
    return tools.add_measurement_calipers(ax, start_time, end_time, label=label)


if __name__ == "__main__":
    # Example usage
    from examples.generate_ecg_data import ECGGenerator
    import matplotlib.pyplot as plt

    # Generate test ECG
    generator = ECGGenerator()
    ecg_data, metadata = generator.generate_normal_sinus_rhythm(duration=10, heart_rate=72)

    # Create measurement tools
    tools = ECGMeasurementTools(metadata['sample_rate'])

    # Test RR interval measurement
    lead_ii = ecg_data[1]  # Lead II
    rr_measurements = tools.measure_rr_intervals(lead_ii, "II")

    print("RR Interval Measurements:")
    for measurement in rr_measurements:
        print(f"  {measurement}")

    # Create measurement report
    report = tools.create_measurements_report(rr_measurements, "Test ECG Analysis")
    print("\n" + report)

    # Create a plot with measurements
    fig, ax = plt.subplots(figsize=(12, 6))

    time_axis = np.linspace(0, 10, len(lead_ii))
    ax.plot(time_axis, lead_ii, 'b-', linewidth=0.8, label='Lead II')

    # Add some example calipers
    tools.add_measurement_calipers(ax, 1.0, 1.8, MeasurementType.RR_INTERVAL)
    tools.add_measurement_calipers(ax, 2.2, 2.4, MeasurementType.QRS_DURATION, color='green')

    # Add measurement grid
    grid_info = tools.add_measurement_grid(ax)

    ax.set_title('ECG with Clinical Measurements')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print("ECG measurements testing complete!")