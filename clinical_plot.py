#!/usr/bin/env python
"""
Clinical ECG Plotting Module
Professional ECG plotting functions that comply with clinical standards.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, FuncFormatter
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings

from ecg_standards import ClinicalECGStandards, ECGDisplayParameters, get_clinical_grid


class ClinicalECGPlotter:
    """Professional clinical ECG plotting with medical standards compliance."""

    def __init__(self,
                 paper_speed: float = 25.0,
                 amplitude_scale: float = 10.0,
                 style: str = 'clinical'):
        """
        Initialize clinical ECG plotter.

        Args:
            paper_speed: Paper speed in mm/s (default: 25.0)
            amplitude_scale: Amplitude scale in mm/mV (default: 10.0)
            style: Plot style ('clinical', 'research', 'print')
        """
        self.standards = ClinicalECGStandards()
        self.paper_speed = paper_speed
        self.amplitude_scale = amplitude_scale
        self.style = style

        # Style configurations
        self.styles = {
            'clinical': {
                'major_grid_color': '#FF0000',  # Red major grid
                'minor_grid_color': '#FFB3B3',  # Light red minor grid
                'signal_color': '#000000',  # Black signal
                'background_color': '#FFFFFF',  # White background
                'calibration_color': '#000000',  # Black calibration
                'text_color': '#000000',  # Black text
                'grid_alpha': 0.8
            },
            'research': {
                'major_grid_color': '#333333',  # Dark gray major grid
                'minor_grid_color': '#CCCCCC',  # Light gray minor grid
                'signal_color': '#0066CC',  # Blue signal
                'background_color': '#FFFFFF',  # White background
                'calibration_color': '#FF6600',  # Orange calibration
                'text_color': '#333333',  # Dark gray text
                'grid_alpha': 0.6
            },
            'print': {
                'major_grid_color': '#000000',  # Black major grid
                'minor_grid_color': '#666666',  # Gray minor grid
                'signal_color': '#000000',  # Black signal
                'background_color': '#FFFFFF',  # White background
                'calibration_color': '#000000',  # Black calibration
                'text_color': '#000000',  # Black text
                'grid_alpha': 1.0
            }
        }

        self.current_style = self.styles.get(style, self.styles['clinical'])

    def plot_12_lead_clinical(self,
                              ecg_data: np.ndarray,
                              sample_rate: int,
                              lead_names: Optional[List[str]] = None,
                              layout: str = '4x3_standard',
                              title: Optional[str] = None,
                              patient_info: Optional[Dict] = None,
                              show_calibration: bool = True,
                              show_measurements: bool = True,
                              show_grid: bool = True,
                              figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Plot 12-lead ECG with clinical standards.

        Args:
            ecg_data: ECG signal array (12, samples)
            sample_rate: Sampling rate in Hz
            lead_names: Lead names (default: standard 12-lead)
            layout: Clinical layout type
            title: Plot title
            patient_info: Patient information dictionary
            show_calibration: Show calibration signal
            show_measurements: Show basic measurements
            show_grid: Show clinical grid
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        if lead_names is None:
            lead_names = self.standards.STANDARD_12_LEADS

        # Get clinical layout
        layout_info = self.standards.get_clinical_lead_layout(layout)

        # Calculate figure size if not provided
        if figsize is None:
            # Standard clinical ECG paper size approximately
            figsize = (11.7, 8.3)  # A4 landscape proportions

        # Create figure and subplots
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(self.current_style['background_color'])

        # Add title and patient info
        if title or patient_info:
            self._add_header(fig, title, patient_info, sample_rate)

        # Convert ECG to clinical scale
        scaled_ecg, conversion_info = self.standards.convert_to_clinical_scale(
            ecg_data, sample_rate, self.paper_speed, self.amplitude_scale
        )

        # Calculate subplot layout
        rows = layout_info['rows']
        cols = layout_info['cols']

        # Create subplots
        gs = fig.add_gridspec(rows, cols,
                              hspace=0.1, wspace=0.1,
                              left=0.08, right=0.95,
                              top=0.85, bottom=0.15)

        # Plot each lead
        for row_idx in range(rows):
            for col_idx in range(cols):
                if row_idx < len(layout_info['layout']) and col_idx < len(layout_info['layout'][row_idx]):
                    lead_name = layout_info['layout'][row_idx][col_idx]

                    if lead_name in lead_names:
                        lead_idx = lead_names.index(lead_name)
                        ax = fig.add_subplot(gs[row_idx, col_idx])

                        # Special handling for rhythm strip (last row in 4x3)
                        if layout == '4x3_standard' and row_idx == rows - 1:
                            duration = min(10.0, scaled_ecg.shape[1] / sample_rate)  # Longer rhythm strip
                        else:
                            duration = min(2.5, scaled_ecg.shape[1] / sample_rate)  # Standard lead duration

                        self._plot_single_lead(
                            ax, scaled_ecg[lead_idx], sample_rate, lead_name,
                            duration=duration,
                            show_grid=show_grid,
                            show_calibration=show_calibration and (row_idx == 0 and col_idx == 0)
                        )

        # Add measurements if requested
        if show_measurements:
            self._add_measurements_panel(fig, scaled_ecg, sample_rate, conversion_info)

        # Add footer with technical parameters
        self._add_footer(fig, conversion_info)

        plt.tight_layout()
        return fig

    def plot_single_lead_clinical(self,
                                  ecg_signal: np.ndarray,
                                  sample_rate: int,
                                  lead_name: str = "ECG",
                                  duration: Optional[float] = None,
                                  title: Optional[str] = None,
                                  show_calibration: bool = True,
                                  show_measurements: bool = True,
                                  show_grid: bool = True,
                                  figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
        """
        Plot single ECG lead with clinical standards.

        Args:
            ecg_signal: ECG signal array (samples,)
            sample_rate: Sampling rate in Hz
            lead_name: Name of the ECG lead
            duration: Duration to plot (None for full signal)
            title: Plot title
            show_calibration: Show calibration signal
            show_measurements: Show measurements
            show_grid: Show clinical grid
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.current_style['background_color'])

        # Convert to clinical scale
        scaled_signal = ecg_signal * (self.amplitude_scale / 10.0)

        if duration is None:
            duration = len(ecg_signal) / sample_rate

        # Plot the signal
        self._plot_single_lead(
            ax, scaled_signal, sample_rate, lead_name,
            duration=duration,
            show_grid=show_grid,
            show_calibration=show_calibration
        )

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold',
                         color=self.current_style['text_color'])

        # Add measurements if requested
        if show_measurements:
            self._add_single_lead_measurements(ax, scaled_signal, sample_rate)

        plt.tight_layout()
        return fig

    def _plot_single_lead(self,
                          ax: plt.Axes,
                          signal: np.ndarray,
                          sample_rate: int,
                          lead_name: str,
                          duration: float,
                          show_grid: bool = True,
                          show_calibration: bool = False):
        """Plot a single ECG lead with clinical formatting."""

        # Calculate time axis
        n_samples = min(len(signal), int(duration * sample_rate))
        time_axis = np.linspace(0, n_samples / sample_rate, n_samples)
        signal_segment = signal[:n_samples]

        # Add calibration signal if requested
        if show_calibration:
            cal_signal, cal_info = self.standards.create_calibration_signal(sample_rate)
            # Prepend calibration to signal
            time_axis = np.linspace(-cal_info['duration_s'], duration - cal_info['duration_s'],
                                    len(cal_signal) + n_samples)
            combined_signal = np.concatenate([cal_signal, signal_segment])
            ax.plot(time_axis, combined_signal,
                    color=self.current_style['signal_color'], linewidth=0.8)
        else:
            ax.plot(time_axis, signal_segment,
                    color=self.current_style['signal_color'], linewidth=0.8)

        # Set up clinical grid
        if show_grid:
            amplitude_range = (np.min(signal_segment), np.max(signal_segment))
            self._setup_clinical_grid(ax, duration, amplitude_range)

        # Format axes
        ax.set_ylabel(f'{lead_name}', fontsize=10, fontweight='bold',
                      color=self.current_style['text_color'])
        ax.set_xlabel('Time (s)', fontsize=8, color=self.current_style['text_color'])

        # Set clinical appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.current_style['text_color'])
        ax.spines['bottom'].set_color(self.current_style['text_color'])
        ax.tick_params(colors=self.current_style['text_color'], labelsize=8)

    def _setup_clinical_grid(self,
                             ax: plt.Axes,
                             duration: float,
                             amplitude_range: Tuple[float, float]):
        """Set up clinical ECG grid on axes."""

        # Get grid parameters
        grid_params = get_clinical_grid(duration, amplitude_range)

        # Set axis limits
        ax.set_xlim(grid_params['time_range'])
        ax.set_ylim(grid_params['voltage_range'])

        # Major grid
        ax.set_xticks(grid_params['major_time_ticks'])
        ax.set_yticks(grid_params['major_voltage_ticks'])
        ax.grid(True, which='major',
                color=self.current_style['major_grid_color'],
                linewidth=0.8, alpha=self.current_style['grid_alpha'])

        # Minor grid
        ax.set_xticks(grid_params['minor_time_ticks'], minor=True)
        ax.set_yticks(grid_params['minor_voltage_ticks'], minor=True)
        ax.grid(True, which='minor',
                color=self.current_style['minor_grid_color'],
                linewidth=0.3, alpha=self.current_style['grid_alpha'])

        # Format tick labels
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}' if x % 1 == 0 or x % 0.2 == 0 else ''))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}' if x != 0 else '0'))

    def _add_header(self,
                    fig: plt.Figure,
                    title: Optional[str],
                    patient_info: Optional[Dict],
                    sample_rate: int):
        """Add clinical header with title and patient information."""

        # Main title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95,
                         color=self.current_style['text_color'])

        # Patient information panel
        if patient_info:
            header_text = []

            # Patient demographics
            if 'patient_name' in patient_info:
                header_text.append(f"Patient: {patient_info['patient_name']}")
            if 'patient_id' in patient_info:
                header_text.append(f"ID: {patient_info['patient_id']}")
            if 'age' in patient_info and 'sex' in patient_info:
                header_text.append(f"Age: {patient_info['age']} Sex: {patient_info['sex']}")

            # Acquisition info
            if 'date' in patient_info:
                header_text.append(f"Date: {patient_info['date']}")
            else:
                header_text.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # Technical parameters
            header_text.append(f"Speed: {self.paper_speed} mm/s")
            header_text.append(f"Gain: {self.amplitude_scale} mm/mV")
            header_text.append(f"Sample Rate: {sample_rate} Hz")

            # Add text box
            textstr = ' | '.join(header_text)
            fig.text(0.5, 0.92, textstr, ha='center', va='top',
                     fontsize=9, color=self.current_style['text_color'],
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    def _add_measurements_panel(self,
                                fig: plt.Figure,
                                ecg_data: np.ndarray,
                                sample_rate: int,
                                conversion_info: Dict):
        """Add basic measurements panel."""

        # Calculate basic measurements
        measurements = self._calculate_basic_measurements(ecg_data, sample_rate)

        # Create measurements text
        meas_text = []
        if measurements['heart_rate'] is not None:
            meas_text.append(f"HR: {measurements['heart_rate']:.0f} bpm")

        meas_text.append(f"Speed: {self.paper_speed} mm/s")
        meas_text.append(f"Scale: {self.amplitude_scale} mm/mV")

        # Add measurements panel
        if meas_text:
            textstr = ' | '.join(meas_text)
            fig.text(0.5, 0.05, textstr, ha='center', va='bottom',
                     fontsize=8, color=self.current_style['text_color'],
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def _add_footer(self, fig: plt.Figure, conversion_info: Dict):
        """Add technical footer information."""

        footer_text = f"Clinical Standards: {self.paper_speed} mm/s, {self.amplitude_scale} mm/mV | " \
                      f"Grid: Major={conversion_info['grid_major_time_s']:.1f}s, {conversion_info['grid_major_voltage_mv']:.1f}mV"

        fig.text(0.02, 0.02, footer_text, ha='left', va='bottom',
                 fontsize=6, color=self.current_style['text_color'], alpha=0.7)

    def _calculate_basic_measurements(self, ecg_data: np.ndarray, sample_rate: int) -> Dict:
        """Calculate basic ECG measurements."""
        measurements = {
            'heart_rate': None,
            'signal_duration': ecg_data.shape[1] / sample_rate,
            'amplitude_range': (np.min(ecg_data), np.max(ecg_data))
        }

        try:
            # Simple heart rate calculation using Lead II (index 1) if available
            lead_for_hr = ecg_data[1] if ecg_data.shape[0] > 1 else ecg_data[0]

            # Find peaks (simplified R-wave detection)
            # This is a basic implementation - Stage 3 will have proper algorithms
            signal_std = np.std(lead_for_hr)
            threshold = np.mean(lead_for_hr) + 2 * signal_std

            # Find peaks above threshold with minimum distance
            peaks = []
            min_distance = int(0.4 * sample_rate)  # Minimum 0.4s between beats

            for i in range(min_distance, len(lead_for_hr) - min_distance):
                if (lead_for_hr[i] > threshold and
                        lead_for_hr[i] > lead_for_hr[i - 1] and
                        lead_for_hr[i] > lead_for_hr[i + 1]):
                    if not peaks or i - peaks[-1] > min_distance:
                        peaks.append(i)

            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / sample_rate
                heart_rate = 60 / np.mean(rr_intervals)
                if 40 <= heart_rate <= 200:  # Reasonable HR range
                    measurements['heart_rate'] = heart_rate

        except Exception as e:
            warnings.warn(f"Could not calculate heart rate: {e}")

        return measurements

    def _add_single_lead_measurements(self, ax: plt.Axes, signal: np.ndarray, sample_rate: int):
        """Add measurements for single lead plot."""
        measurements = self._calculate_basic_measurements(signal.reshape(1, -1), sample_rate)

        if measurements['heart_rate'] is not None:
            ax.text(0.02, 0.98, f"HR: {measurements['heart_rate']:.0f} bpm",
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=10, fontweight='bold')


# Convenience functions for quick access
def plot_clinical_12lead(ecg_data: np.ndarray,
                         sample_rate: int,
                         title: Optional[str] = None,
                         **kwargs) -> plt.Figure:
    """Quick function for clinical 12-lead ECG plotting."""
    plotter = ClinicalECGPlotter()
    return plotter.plot_12_lead_clinical(ecg_data, sample_rate, title=title, **kwargs)


def plot_clinical_single(ecg_signal: np.ndarray,
                         sample_rate: int,
                         lead_name: str = "ECG",
                         **kwargs) -> plt.Figure:
    """Quick function for clinical single-lead ECG plotting."""
    plotter = ClinicalECGPlotter()
    return plotter.plot_single_lead_clinical(ecg_signal, sample_rate, lead_name, **kwargs)


def show_clinical_styles():
    """Display available clinical plotting styles."""
    plotter = ClinicalECGPlotter()
    print("Available Clinical Styles:")
    for style_name, style_config in plotter.styles.items():
        print(f"\n{style_name.upper()}:")
        for param, value in style_config.items():
            print(f"  {param}: {value}")


if __name__ == "__main__":
    # Example usage
    from examples.generate_ecg_data import ECGGenerator

    # Generate test ECG
    generator = ECGGenerator()
    ecg_data, metadata = generator.generate_normal_sinus_rhythm(duration=10, heart_rate=75)

    print(f"Generated ECG: {ecg_data.shape}")

    # Test clinical plotting
    plotter = ClinicalECGPlotter(style='clinical')

    # Plot 12-lead clinical ECG
    fig_12lead = plotter.plot_12_lead_clinical(
        ecg_data,
        metadata['sample_rate'],
        title="Clinical 12-Lead ECG - Stage 2 Test",
        patient_info={
            'patient_name': 'Test Patient',
            'patient_id': 'TEST001',
            'age': '45',
            'sex': 'M'
        }
    )

    # Plot single lead
    fig_single = plotter.plot_single_lead_clinical(
        ecg_data[1],
        metadata['sample_rate'],
        lead_name='Lead II',
        title="Clinical Single Lead - Lead II"
    )

    plt.show()
    print("Clinical plotting test complete!")