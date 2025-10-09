#!/usr/bin/env python
"""
Clinical Report Generation Module
Automated ECG interpretation reports following clinical standards
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ecg_io import ECGMetadata
from signal_processing import ECGSignalProcessor
from ecg_detection import ECGWaveDetector, ECGWaveFeatures
from advanced_analysis import HRVAnalyzer, ArrhythmiaDetector, RhythmType, HRVMetrics
from clinical_plot import ClinicalECGPlotter
from ecg_measurements import ECGMeasurementTools, MeasurementType


@dataclass
class ClinicalFindings:
    """Clinical findings from ECG analysis."""
    # Basic measurements
    heart_rate: float
    rhythm: str
    rhythm_confidence: float

    # Intervals (in ms)
    pr_interval: Optional[float] = None
    qrs_duration: Optional[float] = None
    qt_interval: Optional[float] = None
    qtc_interval: Optional[float] = None

    # HRV metrics
    hrv_metrics: Optional[HRVMetrics] = None

    # Findings and interpretations
    findings: List[str] = None
    abnormalities: List[str] = None
    clinical_significance: str = "Normal"

    # Quality
    signal_quality: str = "Good"
    interpretation_confidence: float = 0.9

    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.abnormalities is None:
            self.abnormalities = []


class ClinicalReportGenerator:
    """Generate comprehensive clinical ECG reports."""

    def __init__(self, sample_rate: int):
        """Initialize report generator."""
        self.sample_rate = sample_rate
        self.processor = ECGSignalProcessor(sample_rate)
        self.detector = ECGWaveDetector(sample_rate)
        self.hrv_analyzer = HRVAnalyzer(sample_rate)
        self.arr_detector = ArrhythmiaDetector(sample_rate)
        self.plotter = ClinicalECGPlotter()

    def analyze_ecg(self, ecg_data: np.ndarray, metadata: ECGMetadata) -> ClinicalFindings:
        """
        Complete ECG analysis for report generation.

        Args:
            ecg_data: ECG signal array (leads, samples)
            metadata: ECG metadata

        Returns:
            ClinicalFindings object
        """
        # Use Lead II for rhythm analysis
        lead_ii = ecg_data[1] if ecg_data.shape[0] > 1 else ecg_data[0]

        # Signal quality assessment
        quality_metrics = self.processor.assess_signal_quality(lead_ii)
        signal_quality = quality_metrics['quality_class']

        # Preprocess
        preprocessed, _ = self.processor.preprocess_ecg(lead_ii)

        # R-peak detection
        r_peaks = self.processor.detect_r_peaks(preprocessed)
        avg_hr, inst_hr = self.processor.calculate_heart_rate(r_peaks)

        # Rhythm detection
        rhythm_type, rhythm_analysis = self.arr_detector.detect_rhythm(lead_ii, r_peaks)

        # Wave detection and interval measurements
        features_list = self.detector.detect_all_waves(lead_ii)

        intervals = {'PR': None, 'QRS': None, 'QT': None, 'QTc': None}
        if features_list:
            # Average intervals from all beats
            pr_intervals = []
            qrs_durations = []
            qt_intervals = []
            qtc_intervals = []

            for features in features_list:
                feat_intervals = self.detector.calculate_intervals_ms(features)
                if feat_intervals['PR']:
                    pr_intervals.append(feat_intervals['PR'])
                if feat_intervals['QRS']:
                    qrs_durations.append(feat_intervals['QRS'])
                if feat_intervals['QT']:
                    qt_intervals.append(feat_intervals['QT'])
                if feat_intervals['QTc']:
                    qtc_intervals.append(feat_intervals['QTc'])

            if pr_intervals:
                intervals['PR'] = np.mean(pr_intervals)
            if qrs_durations:
                intervals['QRS'] = np.mean(qrs_durations)
            if qt_intervals:
                intervals['QT'] = np.mean(qt_intervals)
            if qtc_intervals:
                intervals['QTc'] = np.mean(qtc_intervals)

        # HRV analysis (if sufficient data)
        hrv_metrics = None
        if len(r_peaks) > 10:
            rr_intervals = np.diff(r_peaks) / self.sample_rate
            hrv_metrics = self.hrv_analyzer.analyze_hrv(rr_intervals)

        # Generate findings
        findings, abnormalities, clinical_significance = self._interpret_findings(
            avg_hr, rhythm_type, intervals, hrv_metrics, quality_metrics
        )

        # Create findings object
        clinical_findings = ClinicalFindings(
            heart_rate=avg_hr,
            rhythm=rhythm_type.value,
            rhythm_confidence=rhythm_analysis['confidence'],
            pr_interval=intervals['PR'],
            qrs_duration=intervals['QRS'],
            qt_interval=intervals['QT'],
            qtc_interval=intervals['QTc'],
            hrv_metrics=hrv_metrics,
            findings=findings,
            abnormalities=abnormalities,
            clinical_significance=clinical_significance,
            signal_quality=signal_quality,
            interpretation_confidence=rhythm_analysis['confidence']
        )

        return clinical_findings

    def _interpret_findings(self,
                            hr: float,
                            rhythm: RhythmType,
                            intervals: Dict,
                            hrv_metrics: Optional[HRVMetrics],
                            quality_metrics: Dict) -> Tuple[List[str], List[str], str]:
        """Interpret ECG findings clinically."""
        findings = []
        abnormalities = []

        # Heart rate interpretation
        if 60 <= hr <= 100:
            findings.append(f"Normal heart rate: {hr:.0f} bpm")
        elif hr < 60:
            findings.append(f"Bradycardia: {hr:.0f} bpm")
            abnormalities.append("Sinus bradycardia")
        else:
            findings.append(f"Tachycardia: {hr:.0f} bpm")
            abnormalities.append("Sinus tachycardia")

        # Rhythm interpretation
        if rhythm == RhythmType.NORMAL_SINUS:
            findings.append("Regular sinus rhythm")
        elif rhythm == RhythmType.ATRIAL_FIB:
            findings.append("Irregular rhythm consistent with atrial fibrillation")
            abnormalities.append("Atrial fibrillation")
        elif rhythm == RhythmType.VTACH:
            findings.append("Wide complex tachycardia, possible ventricular tachycardia")
            abnormalities.append("Ventricular tachycardia (possible)")
        else:
            findings.append(f"Rhythm: {rhythm.value}")

        # PR interval
        if intervals['PR']:
            pr = intervals['PR']
            if 120 <= pr <= 200:
                findings.append(f"Normal PR interval: {pr:.0f} ms")
            elif pr > 200:
                findings.append(f"Prolonged PR interval: {pr:.0f} ms")
                abnormalities.append("First-degree AV block")
            else:
                findings.append(f"Short PR interval: {pr:.0f} ms")
                abnormalities.append("Short PR interval (consider pre-excitation)")

        # QRS duration
        if intervals['QRS']:
            qrs = intervals['QRS']
            if qrs <= 120:
                findings.append(f"Normal QRS duration: {qrs:.0f} ms")
            else:
                findings.append(f"Prolonged QRS duration: {qrs:.0f} ms")
                abnormalities.append("Wide QRS complex (consider bundle branch block)")

        # QT interval
        if intervals['QTc']:
            qtc = intervals['QTc']
            if qtc <= 450:
                findings.append(f"Normal QTc: {qtc:.0f} ms")
            elif 450 < qtc <= 470:
                findings.append(f"Borderline QTc: {qtc:.0f} ms")
                abnormalities.append("Borderline QT prolongation")
            else:
                findings.append(f"Prolonged QTc: {qtc:.0f} ms")
                abnormalities.append("QT prolongation (risk of arrhythmia)")

        # HRV interpretation
        if hrv_metrics:
            findings.append(f"HRV: {hrv_metrics.hrv_category} (SDNN: {hrv_metrics.sdnn:.0f} ms)")
            if hrv_metrics.hrv_category == "Poor":
                abnormalities.append("Reduced heart rate variability")

        # Signal quality
        if quality_metrics['overall_quality'] < 60:
            findings.append(f"Signal quality: {quality_metrics['quality_class']}")
            findings.append("Consider repeat ECG for better quality")

        # Determine clinical significance
        if not abnormalities:
            clinical_significance = "Normal ECG"
        elif any(term in ' '.join(abnormalities).lower() for term in
                 ['ventricular', 'qt prolongation', 'atrial fibrillation']):
            clinical_significance = "Clinically significant abnormalities - Recommend urgent evaluation"
        else:
            clinical_significance = "Minor abnormalities - Clinical correlation recommended"

        return findings, abnormalities, clinical_significance

    def generate_text_report(self, metadata: ECGMetadata, findings: ClinicalFindings) -> str:
        """Generate formatted text report."""
        report = []
        report.append("=" * 80)
        report.append("ELECTROCARDIOGRAM INTERPRETATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Patient information
        report.append("PATIENT INFORMATION:")
        report.append("-" * 80)
        if metadata.patient_name:
            report.append(f"Name: {metadata.patient_name}")
        report.append(f"Patient ID: {metadata.patient_id}")
        if metadata.age:
            report.append(f"Age: {metadata.age} years")
        if metadata.sex:
            report.append(f"Sex: {metadata.sex}")
        report.append(f"Date: {metadata.date}")
        report.append(f"Time: {metadata.time}")
        report.append("")

        # Technical parameters
        report.append("TECHNICAL PARAMETERS:")
        report.append("-" * 80)
        report.append(f"Sample Rate: {metadata.sample_rate} Hz")
        report.append(f"Duration: {metadata.duration:.1f} seconds")
        report.append(f"Leads: {metadata.n_leads} ({', '.join(metadata.lead_names)})")
        report.append(f"Signal Quality: {findings.signal_quality}")
        if metadata.device:
            report.append(f"Device: {metadata.device}")
        report.append("")

        # Measurements
        report.append("MEASUREMENTS:")
        report.append("-" * 80)
        report.append(f"Heart Rate: {findings.heart_rate:.0f} bpm")
        report.append(f"Rhythm: {findings.rhythm}")

        if findings.pr_interval:
            report.append(f"PR Interval: {findings.pr_interval:.0f} ms")
        if findings.qrs_duration:
            report.append(f"QRS Duration: {findings.qrs_duration:.0f} ms")
        if findings.qt_interval:
            report.append(f"QT Interval: {findings.qt_interval:.0f} ms")
        if findings.qtc_interval:
            report.append(f"QTc Interval: {findings.qtc_interval:.0f} ms")

        if findings.hrv_metrics:
            report.append(f"HRV (SDNN): {findings.hrv_metrics.sdnn:.0f} ms ({findings.hrv_metrics.hrv_category})")
        report.append("")

        # Interpretation
        report.append("INTERPRETATION:")
        report.append("-" * 80)

        if findings.findings:
            report.append("Findings:")
            for i, finding in enumerate(findings.findings, 1):
                report.append(f"  {i}. {finding}")
            report.append("")

        if findings.abnormalities:
            report.append("ABNORMALITIES:")
            for i, abnormality in enumerate(findings.abnormalities, 1):
                report.append(f"  {i}. {abnormality}")
            report.append("")

        # Clinical significance
        report.append("CLINICAL SIGNIFICANCE:")
        report.append(f"  {findings.clinical_significance}")
        report.append("")

        # Confidence and disclaimer
        report.append(f"Interpretation Confidence: {findings.interpretation_confidence:.0%}")
        report.append("")
        report.append("DISCLAIMER:")
        report.append("-" * 80)
        report.append("This is an automated ECG interpretation generated by computer algorithm.")
        report.append("All automated interpretations should be verified by a qualified physician.")
        report.append("This report is for research and educational purposes only.")
        report.append("")

        # Signature section
        report.append("REVIEWED BY:")
        report.append("-" * 80)
        if metadata.technician:
            report.append(f"Technician: {metadata.technician}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("=" * 80)

        return '\n'.join(report)

    def generate_pdf_report(self,
                            filepath: str,
                            ecg_data: np.ndarray,
                            metadata: ECGMetadata,
                            findings: ClinicalFindings):
        """
        Generate comprehensive PDF report with ECG plots.

        Args:
            filepath: Output PDF file path
            ecg_data: ECG signal array
            metadata: ECG metadata
            findings: Clinical findings
        """
        with PdfPages(filepath) as pdf:
            # Page 1: ECG Waveforms
            fig = self.plotter.plot_12_lead_clinical(
                ecg_data,
                self.sample_rate,
                title="12-Lead Electrocardiogram",
                patient_info={
                    'patient_name': metadata.patient_name or 'N/A',
                    'patient_id': metadata.patient_id,
                    'age': str(metadata.age) if metadata.age else 'N/A',
                    'sex': metadata.sex or 'N/A',
                    'date': metadata.date
                },
                show_calibration=True,
                show_measurements=True
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 2: Interpretation Report
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')

            # Generate text report
            text_report = self.generate_text_report(metadata, findings)

            # Add text to figure
            ax.text(0.05, 0.95, text_report,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    fontfamily='monospace',
                    wrap=True)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'ECG Interpretation Report'
            d['Author'] = 'ECG Analysis System'
            d['Subject'] = f'Patient: {metadata.patient_id}'
            d['Keywords'] = 'ECG, Electrocardiogram, Clinical Report'
            d['CreationDate'] = datetime.now()


# Convenience functions
def generate_quick_report(ecg_data: np.ndarray,
                          metadata: ECGMetadata,
                          sample_rate: int) -> str:
    """Generate quick text report."""
    generator = ClinicalReportGenerator(sample_rate)
    findings = generator.analyze_ecg(ecg_data, metadata)
    return generator.generate_text_report(metadata, findings)


def generate_full_report(ecg_data: np.ndarray,
                         metadata: ECGMetadata,
                         sample_rate: int,
                         output_path: str = 'ecg_report.pdf'):
    """Generate complete PDF report."""
    generator = ClinicalReportGenerator(sample_rate)
    findings = generator.analyze_ecg(ecg_data, metadata)
    generator.generate_pdf_report(output_path, ecg_data, metadata, findings)
    return findings


if __name__ == "__main__":
    from examples.generate_ecg_data import ECGGenerator
    from ecg_io import ECGMetadata

    print("Testing Clinical Report Generation...")

    # Generate test ECG
    gen = ECGGenerator()
    ecg_data, gen_meta = gen.generate_normal_sinus_rhythm(duration=10, heart_rate=75)

    # Create metadata
    metadata = ECGMetadata(
        patient_id='TEST001',
        patient_name='John Doe',
        age=45,
        sex='M',
        sample_rate=gen_meta['sample_rate'],
        duration=10.0,
        n_leads=12,
        lead_names=gen_meta['leads'],
        device='ECG Simulator v1.0',
        technician='Test Technician'
    )

    # Generate report
    generator = ClinicalReportGenerator(metadata.sample_rate)
    findings = generator.analyze_ecg(ecg_data, metadata)

    # Text report
    text_report = generator.generate_text_report(metadata, findings)
    print(text_report)

    # PDF report
    print("\nGenerating PDF report...")
    generator.generate_pdf_report('test_report.pdf', ecg_data, metadata, findings)
    print("PDF report generated: test_report.pdf")

    print("\nClinical report generation working!")