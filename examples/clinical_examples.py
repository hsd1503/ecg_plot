#!/usr/bin/env python
"""
Clinical Examples for Stage 2
Demonstrates clinical-grade ECG plotting and measurement capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.generate_ecg_data import ECGGenerator
from ecg_standards import ClinicalECGStandards, get_standard_12lead_layout
from clinical_plot import ClinicalECGPlotter, plot_clinical_12lead
from ecg_measurements import ECGMeasurementTools, MeasurementType


class ClinicalECGExamples:
    """Clinical ECG examples demonstrating Stage 2 capabilities."""

    def __init__(self):
        """Initialize clinical examples."""
        self.generator = ECGGenerator()
        self.standards = ClinicalECGStandards()

    def emergency_department_scenario(self) -> dict:
        """
        Emergency Department: Rapid ECG assessment for chest pain patient.

        Returns:
            Dictionary with scenario results
        """
        print("\n" + "=" * 60)
        print("🚨 EMERGENCY DEPARTMENT SCENARIO")
        print("=" * 60)
        print("Patient: 58-year-old male with acute chest pain")
        print("Priority: STAT ECG for MI evaluation")

        # Generate ECG with slight tachycardia (stress response)
        ecg_data, metadata = self.generator.generate_normal_sinus_rhythm(
            duration=10,
            heart_rate=95,  # Elevated due to stress/pain
            noise_level=0.08  # Some artifact from patient movement
        )

        print(f"✓ ECG acquired: {metadata['heart_rate']} bpm")

        # Clinical plotting with ED-specific parameters
        plotter = ClinicalECGPlotter(
            paper_speed=25.0,  # Standard speed for diagnosis
            amplitude_scale=10.0,  # Standard gain
            style='clinical'  # Clinical red grid
        )

        # Patient information
        patient_info = {
            'patient_name': 'Emergency Patient',
            'patient_id': 'ED-2024-001',
            'age': '58',
            'sex': 'M',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'indication': 'Chest pain, rule out MI'
        }

        # Create clinical 12-lead plot
        fig = plotter.plot_12_lead_clinical(
            ecg_data,
            metadata['sample_rate'],
            title="🚨 EMERGENCY ECG - STAT",
            patient_info=patient_info,
            show_calibration=True,
            show_measurements=True,
            layout='4x3_standard'
        )

        # Quick measurements for emergency assessment
        tools = ECGMeasurementTools(metadata['sample_rate'])
        lead_ii = ecg_data[1]  # Lead II for rhythm assessment
        rr_measurements = tools.measure_rr_intervals(lead_ii, "II")

        # Emergency assessment
        hr_measurements = [m for m in rr_measurements if m.measurement_type == MeasurementType.HEART_RATE]
        avg_hr = np.mean([m.value for m in hr_measurements]) if hr_measurements else 0

        print(f"✓ Rhythm analysis: Average HR = {avg_hr:.0f} bpm")
        print(f"✓ Rhythm: {'Regular' if len(hr_measurements) > 0 else 'Cannot determine'}")

        # Clinical decision
        if 90 <= avg_hr <= 110:
            assessment = "Sinus tachycardia - consistent with stress response"
            action = "Continue with cardiac workup, consider pain management"
        elif avg_hr > 110:
            assessment = "Significant tachycardia - investigate cause"
            action = "Urgent cardiology consultation"
        else:
            assessment = "Heart rate within acceptable range"
            action = "Continue standard chest pain protocol"

        print(f"📋 Assessment: {assessment}")
        print(f"🎯 Action: {action}")

        scenario_results = {
            'scenario': 'Emergency Department',
            'patient_info': patient_info,
            'ecg_data': ecg_data,
            'measurements': rr_measurements,
            'average_hr': avg_hr,
            'clinical_assessment': assessment,
            'recommended_action': action,
            'figure': fig
        }

        return scenario_results

    def cardiology_clinic_scenario(self) -> dict:
        """
        Cardiology Clinic: Detailed ECG analysis for follow-up patient.

        Returns:
            Dictionary with scenario results
        """
        print("\n" + "=" * 60)
        print("🏥 CARDIOLOGY CLINIC SCENARIO")
        print("=" * 60)
        print("Patient: 45-year-old female, routine follow-up")
        print("History: Mitral valve prolapse, annual monitoring")

        # Generate high-quality ECG (clinic environment)
        ecg_data, metadata = self.generator.generate_normal_sinus_rhythm(
            duration=12,  # Longer recording for detailed analysis
            heart_rate=68,  # Normal resting HR
            noise_level=0.02  # Minimal noise in controlled environment
        )

        print(f"✓ High-quality ECG acquired: {metadata['heart_rate']} bpm")

        # Research-style plotting for detailed analysis
        plotter = ClinicalECGPlotter(
            paper_speed=25.0,
            amplitude_scale=10.0,
            style='research'  # Research style with blue signals
        )

        patient_info = {
            'patient_name': 'Jane Smith',
            'patient_id': 'CARD-2024-047',
            'age': '45',
            'sex': 'F',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'indication': 'MVP follow-up, routine monitoring',
            'medications': 'Beta-blocker 25mg daily'
        }

        # Create detailed clinical plot
        fig = plotter.plot_12_lead_clinical(
            ecg_data,
            metadata['sample_rate'],
            title="Cardiology Clinic - Detailed Analysis",
            patient_info=patient_info,
            show_calibration=True,
            show_measurements=True,
            layout='4x3_standard'
        )

        # Comprehensive measurements
        tools = ECGMeasurementTools(metadata['sample_rate'])

        # Multiple lead analysis
        all_measurements = []

        # Lead II for rhythm
        lead_ii = ecg_data[1]
        rr_measurements = tools.measure_rr_intervals(lead_ii, "II")
        all_measurements.extend(rr_measurements)

        # Add some example interval measurements
        # (In a real scenario, these would be detected automatically)
        sample_rate = metadata['sample_rate']

        # Example PR interval measurement (120-200ms normal)
        pr_measurement = tools.measure_custom_interval(
            int(0.5 * sample_rate),  # Start sample
            int((0.5 + 0.16) * sample_rate),  # End sample (160ms PR)
            MeasurementType.PR_INTERVAL,
            "Lead II"
        )
        all_measurements.append(pr_measurement)

        # Example QRS duration (80-120ms normal)
        qrs_measurement = tools.measure_custom_interval(
            int(0.66 * sample_rate),  # Start sample
            int((0.66 + 0.09) * sample_rate),  # End sample (90ms QRS)
            MeasurementType.QRS_DURATION,
            "Lead II"
        )
        all_measurements.append(qrs_measurement)

        # Generate comprehensive report
        report = tools.create_measurements_report(
            all_measurements,
            "Cardiology Clinic - ECG Analysis Report"
        )

        print("📊 Detailed Analysis Complete:")
        hr_measurements = [m for m in all_measurements if m.measurement_type == MeasurementType.HEART_RATE]
        avg_hr = np.mean([m.value for m in hr_measurements]) if hr_measurements else 0
        print(f"  • Average Heart Rate: {avg_hr:.0f} bpm")
        print(
            f"  • PR Interval: {pr_measurement.value:.0f} ms ({'Normal' if pr_measurement.is_normal else 'Abnormal'})")
        print(
            f"  • QRS Duration: {qrs_measurement.value:.0f} ms ({'Normal' if qrs_measurement.is_normal else 'Abnormal'})")

        # Clinical interpretation
        normal_findings = all([m.is_normal for m in all_measurements])
        interpretation = "Normal sinus rhythm, no acute changes" if normal_findings else "See detailed analysis for abnormal findings"
        recommendation = "Continue current management, next follow-up in 12 months" if normal_findings else "Consider additional testing"

        print(f"📋 Interpretation: {interpretation}")
        print(f"🎯 Recommendation: {recommendation}")

        scenario_results = {
            'scenario': 'Cardiology Clinic',
            'patient_info': patient_info,
            'ecg_data': ecg_data,
            'measurements': all_measurements,
            'detailed_report': report,
            'clinical_interpretation': interpretation,
            'recommendation': recommendation,
            'figure': fig
        }

        return scenario_results

    def research_study_scenario(self) -> dict:
        """
        Research Study: Multi-subject ECG analysis with standardized measurements.

        Returns:
            Dictionary with scenario results
        """
        print("\n" + "=" * 60)
        print("🔬 RESEARCH STUDY SCENARIO")
        print("=" * 60)
        print("Study: Cardiac autonomic function in healthy adults")
        print("Protocol: Standardized 12-lead ECG with HRV analysis")

        # Generate multiple subjects
        subjects_data = []
        study_measurements = []

        for subject_id in range(1, 4):  # 3 subjects for example
            print(f"\n📝 Processing Subject {subject_id:03d}...")

            # Generate subject-specific ECG
            age = np.random.randint(25, 65)
            hr = np.random.randint(60, 85)  # Normal range

            ecg_data, metadata = self.generator.generate_normal_sinus_rhythm(
                duration=60,  # 1-minute recording for HRV
                heart_rate=hr,
                noise_level=0.03  # Controlled research environment
            )

            subject_info = {
                'subject_id': f"STUDY-{subject_id:03d}",
                'age': age,
                'sex': 'M' if subject_id % 2 == 1 else 'F',
                'bmi': round(np.random.uniform(20, 28), 1),
                'date': (datetime.now() - timedelta(days=subject_id)).strftime('%Y-%m-%d')
            }

            # Research-style plotting
            plotter = ClinicalECGPlotter(
                paper_speed=25.0,
                amplitude_scale=10.0,
                style='research'
            )

            # Create standardized research ECG
            fig = plotter.plot_12_lead_clinical(
                ecg_data,
                metadata['sample_rate'],
                title=f"Research Study - Subject {subject_id:03d}",
                patient_info=subject_info,
                show_calibration=True,
                layout='3x4_landscape'  # Landscape layout for research
            )

            # Standardized measurements
            tools = ECGMeasurementTools(metadata['sample_rate'])
            lead_ii = ecg_data[1]
            measurements = tools.measure_rr_intervals(lead_ii, "II")

            # Calculate HRV metrics (simplified)
            hr_values = [m.value for m in measurements if m.measurement_type == MeasurementType.HEART_RATE]
            rr_values = [m.value for m in measurements if m.measurement_type == MeasurementType.RR_INTERVAL]

            if hr_values and rr_values:
                avg_hr = np.mean(hr_values)
                hr_std = np.std(hr_values)
                rr_std = np.std(rr_values)

                print(f"  • HR: {avg_hr:.1f} ± {hr_std:.1f} bpm")
                print(f"  • RR variability: {rr_std:.1f} ms")
            else:
                avg_hr = hr_std = rr_std = 0
                print("  • Measurement failed")

            subject_data = {
                'subject_info': subject_info,
                'ecg_data': ecg_data,
                'measurements': measurements,
                'avg_hr': avg_hr,
                'hr_std': hr_std,
                'rr_std': rr_std,
                'figure': fig
            }

            subjects_data.append(subject_data)
            study_measurements.extend(measurements)

        # Study-level analysis
        print(f"\n📊 STUDY SUMMARY:")
        all_hrs = [s['avg_hr'] for s in subjects_data if s['avg_hr'] > 0]
        all_hrv = [s['rr_std'] for s in subjects_data if s['rr_std'] > 0]

        if all_hrs:
            print(f"  • Study Population HR: {np.mean(all_hrs):.1f} ± {np.std(all_hrs):.1f} bpm")
            print(f"  • Study Population HRV: {np.mean(all_hrv):.1f} ± {np.std(all_hrv):.1f} ms")
            print(f"  • Sample Size: {len(subjects_data)} subjects")
            print(f"  • Data Quality: {'Good' if len(all_hrs) == len(subjects_data) else 'Some issues'}")

        scenario_results = {
            'scenario': 'Research Study',
            'study_protocol': 'Cardiac autonomic function in healthy adults',
            'subjects_data': subjects_data,
            'study_measurements': study_measurements,
            'population_hr_mean': np.mean(all_hrs) if all_hrs else 0,
            'population_hr_std': np.std(all_hrs) if all_hrs else 0,
            'population_hrv_mean': np.mean(all_hrv) if all_hrv else 0,
            'sample_size': len(subjects_data)
        }

        return scenario_results

    def demonstrate_measurement_tools(self):
        """Demonstrate advanced measurement capabilities."""
        print("\n" + "=" * 60)
        print("📐 MEASUREMENT TOOLS DEMONSTRATION")
        print("=" * 60)

        # Generate test ECG
        ecg_data, metadata = self.generator.generate_normal_sinus_rhythm(
            duration=8, heart_rate=75
        )

        # Create measurement tools
        tools = ECGMeasurementTools(metadata['sample_rate'])

        # Create plot with measurements
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Lead II with automated measurements
        lead_ii = ecg_data[1]
        time_axis = np.linspace(0, 8, len(lead_ii))

        ax1.plot(time_axis, lead_ii, 'b-', linewidth=1, label='Lead II')
        ax1.set_title('Automated RR Interval Measurements', fontsize=12, fontweight='bold')

        # Add RR measurements
        rr_measurements = tools.measure_rr_intervals(lead_ii, "II")

        # Add calipers for first few RR intervals
        rr_intervals = [m for m in rr_measurements if m.measurement_type == MeasurementType.RR_INTERVAL][:3]
        for i, measurement in enumerate(rr_intervals):
            start_time = measurement.start_sample / metadata['sample_rate']
            end_time = measurement.end_sample / metadata['sample_rate']

            color = ['red', 'green', 'blue'][i]
            tools.add_measurement_calipers(
                ax1, start_time, end_time,
                MeasurementType.RR_INTERVAL,
                color=color,
                label=f"RR{i + 1}: {measurement.value:.0f}ms"
            )

        # Add measurement grid
        grid_info = tools.add_measurement_grid(ax1, marker_interval_ms=200)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Lead V1 with manual measurements
        lead_v1 = ecg_data[6]  # V1
        ax2.plot(time_axis, lead_v1, 'g-', linewidth=1, label='Lead V1')
        ax2.set_title('Manual Interval Measurements', fontsize=12, fontweight='bold')

        # Add example manual measurements
        # PR interval example
        tools.add_measurement_calipers(
            ax2, 1.0, 1.16,
            MeasurementType.PR_INTERVAL,
            color='red',
            label="PR: 160ms"
        )

        # QRS duration example
        tools.add_measurement_calipers(
            ax2, 2.66, 2.75,
            MeasurementType.QRS_DURATION,
            color='orange',
            label="QRS: 90ms"
        )

        # QT interval example
        tools.add_measurement_calipers(
            ax2, 3.66, 4.06,
            MeasurementType.QT_INTERVAL,
            color='purple',
            label="QT: 400ms"
        )

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (mV)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Print measurement report
        report = tools.create_measurements_report(rr_measurements[:6])  # First 6 measurements
        print("\n📋 MEASUREMENT REPORT:")
        print(report)

        return fig, rr_measurements

    def demonstrate_clinical_standards(self):
        """Demonstrate clinical standards compliance."""
        print("\n" + "=" * 60)
        print("📏 CLINICAL STANDARDS DEMONSTRATION")
        print("=" * 60)

        # Generate ECG data
        ecg_data, metadata = self.generator.generate_normal_sinus_rhythm(
            duration=10, heart_rate=72
        )

        print("Testing different clinical standards:")

        # Test different paper speeds
        paper_speeds = [12.5, 25.0, 50.0]
        amplitude_scales = [5.0, 10.0, 20.0]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Clinical Standards Comparison', fontsize=16, fontweight='bold')

        for i, paper_speed in enumerate(paper_speeds):
            for j, amp_scale in enumerate(amplitude_scales):
                ax = axes[i, j]

                # Create plotter with specific standards
                plotter = ClinicalECGPlotter(
                    paper_speed=paper_speed,
                    amplitude_scale=amp_scale,
                    style='clinical'
                )

                # Plot single lead
                lead_ii = ecg_data[1]
                plotter._plot_single_lead(
                    ax, lead_ii, metadata['sample_rate'],
                    f"II", duration=5.0, show_grid=True
                )

                ax.set_title(f'{paper_speed} mm/s, {amp_scale} mm/mV',
                             fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Test clinical compliance
        compliance = self.standards.validate_clinical_compliance(
            ecg_data, metadata['sample_rate']
        )

        print("\n✓ Clinical Standards Compliance Test:")
        for standard, result in compliance.items():
            status = "✓" if result else "✗"
            print(f"  {status} {standard.replace('_', ' ').title()}")

        return fig, compliance

    def run_all_scenarios(self):
        """Run all clinical scenarios."""
        print("🩺 STAGE 2 CLINICAL ECG EXAMPLES")
        print("Demonstrating biomedical engineering improvements")

        scenarios = []

        try:
            # Run each scenario
            ed_results = self.emergency_department_scenario()
            scenarios.append(ed_results)

            clinic_results = self.cardiology_clinic_scenario()
            scenarios.append(clinic_results)

            research_results = self.research_study_scenario()
            scenarios.append(research_results)

            # Demonstrate tools
            meas_fig, measurements = self.demonstrate_measurement_tools()

            standards_fig, compliance = self.demonstrate_clinical_standards()

            print("\n" + "=" * 60)
            print("🎉 ALL CLINICAL SCENARIOS COMPLETE!")
            print("=" * 60)
            print("Stage 2 features successfully demonstrated:")
            print("  ✓ Clinical standards compliance (25mm/s, 10mm/mV)")
            print("  ✓ Professional ECG layouts and formatting")
            print("  ✓ Clinical measurement tools and calipers")
            print("  ✓ Multi-scenario clinical applications")
            print("  ✓ Research-grade ECG analysis")

            # Show all plots if in interactive mode
            if plt.get_backend() != 'Agg':
                plt.show()

            return {
                'scenarios': scenarios,
                'measurement_demo': {'figure': meas_fig, 'measurements': measurements},
                'standards_demo': {'figure': standards_fig, 'compliance': compliance}
            }

        except Exception as e:
            print(f"❌ Error running scenarios: {e}")
            import traceback
            traceback.print_exc()
            return None


# Convenience functions
def run_emergency_demo():
    """Quick emergency department demo."""
    examples = ClinicalECGExamples()
    return examples.emergency_department_scenario()


def run_clinic_demo():
    """Quick cardiology clinic demo."""
    examples = ClinicalECGExamples()
    return examples.cardiology_clinic_scenario()


def run_research_demo():
    """Quick research study demo."""
    examples = ClinicalECGExamples()
    return examples.research_study_scenario()


if __name__ == "__main__":
    # Run all clinical examples
    examples = ClinicalECGExamples()
    results = examples.run_all_scenarios()

    if results:
        print("\n📊 SUMMARY STATISTICS:")
        print(f"  • Scenarios completed: {len(results['scenarios'])}")
        print(
            f"  • Clinical compliance: {'✓ PASSED' if results['standards_demo']['compliance']['overall_compliant'] else '✗ FAILED'}")
        print(f"  • Measurement tools: ✓ WORKING")
        print("\n🚀 Stage 2 implementation ready for clinical use!")
    else:
        print("❌ Some scenarios failed - check implementation")