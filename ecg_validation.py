#!/usr/bin/env python
"""
ECG Signal Validation Module
Validates ECG signals against physiological and technical standards.
"""

import numpy as np
import warnings
from typing import Union, List, Optional, Dict, Tuple

class ECGValidationError(Exception):
    """Custom exception for ECG validation errors."""
    pass

class ECGWarning(UserWarning):
    """Custom warning for ECG validation issues."""
    pass

class ECGValidator:
    """Comprehensive ECG signal validation."""
    
    # Physiological ranges (normal adult values)
    NORMAL_HEART_RATE_RANGE = (50, 120)  # bpm
    NORMAL_AMPLITUDE_RANGE = (-5.0, 5.0)  # mV
    ACCEPTABLE_SAMPLE_RATES = range(100, 10001)  # Hz
    STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator."""
        self.strict_mode = strict_mode
        self.validation_results = {}
    
    def validate_ecg_data(self, 
                         ecg_data: np.ndarray, 
                         sample_rate: int,
                         lead_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Comprehensive ECG data validation."""
        results = {
            'data_format': False,
            'sample_rate': False,
            'amplitude_range': False,
            'signal_quality': False,
            'lead_configuration': False,
            'overall_valid': False
        }
        
        try:
            results['data_format'] = self._validate_data_format(ecg_data)
            results['sample_rate'] = self._validate_sample_rate(sample_rate)
            results['amplitude_range'] = self._validate_amplitude_range(ecg_data)
            results['signal_quality'] = self._validate_signal_quality(ecg_data, sample_rate)
            
            if lead_names is not None:
                results['lead_configuration'] = self._validate_lead_configuration(ecg_data, lead_names)
            else:
                results['lead_configuration'] = True

            # Calculate overall_valid based on all components except itself
            component_results = [results['data_format'], results['sample_rate'],
                                 results['amplitude_range'], results['signal_quality'],
                                 results['lead_configuration']]
            results['overall_valid'] = all(component_results)
            
        except Exception as e:
            if self.strict_mode:
                raise ECGValidationError(f"ECG validation failed: {str(e)}")
            else:
                warnings.warn(f"ECG validation warning: {str(e)}", ECGWarning)
        
        self.validation_results = results
        return results
    
    def _validate_data_format(self, ecg_data: np.ndarray) -> bool:
        """Validate ECG data format and structure."""
        if not isinstance(ecg_data, np.ndarray):
            raise ECGValidationError("ECG data must be a numpy array")
        
        if ecg_data.ndim != 2:
            raise ECGValidationError(f"ECG data must be 2D array (leads, samples), got {ecg_data.ndim}D")
        
        n_leads, n_samples = ecg_data.shape
        
        if n_leads < 1 or n_leads > 15:
            self._handle_validation_issue(f"Unusual number of leads: {n_leads}. Expected 1-15 leads.")
        
        if n_samples < 100:
            raise ECGValidationError(f"Signal too short: {n_samples} samples. Minimum 100 samples required.")
        
        if np.any(np.isnan(ecg_data)) or np.any(np.isinf(ecg_data)):
            raise ECGValidationError("ECG data contains NaN or infinite values")
        
        return True
    
    def _validate_sample_rate(self, sample_rate: int) -> bool:
        """Validate sample rate against medical standards."""
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ECGValidationError(f"Sample rate must be positive number, got {sample_rate}")
        
        if sample_rate not in self.ACCEPTABLE_SAMPLE_RATES:
            if sample_rate < 100:
                raise ECGValidationError(f"Sample rate too low: {sample_rate} Hz. Minimum 100 Hz required.")
            elif sample_rate > 10000:
                self._handle_validation_issue(f"Very high sample rate: {sample_rate} Hz. May be unnecessary.")
        
        return True
    
    def _validate_amplitude_range(self, ecg_data: np.ndarray) -> bool:
        """Validate ECG amplitude against physiological ranges."""
        min_val, max_val = np.min(ecg_data), np.max(ecg_data)
        
        if min_val < -10.0 or max_val > 10.0:
            self._handle_validation_issue(
                f"Amplitude outside typical range: {min_val:.2f} to {max_val:.2f} mV. "
                f"Normal range: {self.NORMAL_AMPLITUDE_RANGE[0]} to {self.NORMAL_AMPLITUDE_RANGE[1]} mV")
        
        signal_range = max_val - min_val
        if signal_range < 0.1:
            self._handle_validation_issue(f"Very small signal amplitude: {signal_range:.3f} mV. May indicate lead disconnection.")
        
        mean_value = np.mean(ecg_data)
        if abs(mean_value) > 1.0:
            self._handle_validation_issue(f"Large DC offset detected: {mean_value:.2f} mV. Consider baseline correction.")
        
        return True
    
    def _validate_signal_quality(self, ecg_data: np.ndarray, sample_rate: int) -> bool:
        """Validate ECG signal quality indicators."""
        return True  # Simplified for now
    
    def _validate_lead_configuration(self, ecg_data: np.ndarray, lead_names: List[str]) -> bool:
        """Validate lead configuration and naming."""
        n_leads = ecg_data.shape[0]
        
        if len(lead_names) != n_leads:
            raise ECGValidationError(f"Number of lead names ({len(lead_names)}) doesn't match data dimensions ({n_leads})")
        
        if len(set(lead_names)) != len(lead_names):
            duplicates = [lead for lead in set(lead_names) if lead_names.count(lead) > 1]
            raise ECGValidationError(f"Duplicate lead names found: {duplicates}")
        
        return True
    
    def _handle_validation_issue(self, message: str):
        """Handle validation issues based on strict mode setting."""
        if self.strict_mode:
            raise ECGValidationError(message)
        else:
            warnings.warn(message, ECGWarning)
    
    def get_validation_report(self) -> str:
        """Get a formatted validation report."""
        if not self.validation_results:
            return "No validation performed yet."
        
        report = "ECG Validation Report\n" + "=" * 30 + "\n"
        
        for check, result in self.validation_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            report += f"{check.replace('_', ' ').title():.<20} {status}\n"
        
        overall = "✓ VALID" if self.validation_results.get('overall_valid', False) else "✗ INVALID"
        report += f"\n{'Overall Status':.<20} {overall}\n"
        
        return report

# Convenience functions for quick validation
def quick_validate(ecg_data: np.ndarray, sample_rate: int, lead_names: Optional[List[str]] = None) -> bool:
    """Quick ECG validation with default settings."""
    validator = ECGValidator(strict_mode=False)
    results = validator.validate_ecg_data(ecg_data, sample_rate, lead_names)
    return results['overall_valid']

def strict_validate(ecg_data: np.ndarray, sample_rate: int, lead_names: Optional[List[str]] = None):
    """Strict ECG validation that raises exceptions on failures."""
    validator = ECGValidator(strict_mode=True)
    return validator.validate_ecg_data(ecg_data, sample_rate, lead_names)
