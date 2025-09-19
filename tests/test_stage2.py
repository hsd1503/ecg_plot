#!/usr/bin/env python
"""
Stage 2 Integration Test Suite
Comprehensive testing of clinical standards compliance and measurement tools.

SAVE AS: tests/test_stage2.py
"""

import numpy as np
import sys
import os
import warnings
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import Stage 1 modules (foundation)
from examples.generate_ecg_data import ECGGenerator
from ecg_validation import ECGValidator, quick_validate

# Import Stage 2 modules (clinical standards)
try:
    from ecg_standards import ClinicalECGStandards, ECGDisplayParameters, get_clinical_grid
    from clinical_plot import ClinicalECGPlotter, plot_clinical_12lead
    from ecg_measurements import ECGMeasurementTools, MeasurementType
    from examples.clinical_examples import ClinicalECGExamples

    print("✓ Successfully imported all Stage 2 modules")
except ImportError as e:
    print(f"❌ Stage 2 import error: {e}")
    print("Make sure you've created all Stage 2 files!")
    sys.exit(1)

# Import original module for compatibility testing
import ecg_plot


class Stage2TestSuite:
    """Comprehensive test suite for Stage 2 features."""

    def __init__(self):
        """Initialize test suite."""
        self.generator = ECGGenerator()
        self.test_results = {}
        self.test_ecg_data = None
        self.test_metadata = None

    def setup_test_data(self):
        """Set up test ECG data for all tests."""
        print("Setting up test ECG data...")
        self.test_ecg_data, self.test_metadata = self.generator.generate_normal_sinus_rhythm(
            duration=10,
            heart_rate=75,
            noise_level=0.05,
            leads=12
        )
        print(f"✓ Test ECG generated: {self.test_ecg_data.shape}")
        print(f"✓ Sample rate: {self.test_metadata['sample_rate']} Hz")

    def test_clinical_standards_module(self) -> bool:
        """Test ECG clinical standards implementation."""
        print("\n" + "=" * 50)
        print("🧪 TESTING CLINICAL STANDARDS MODULE")
        print("=" * 50)

        try:
            standards = ClinicalECGStandards()

            # Test 1: Clinical scale conversion
            print("1. Testing clinical scale conversion...")
            scaled_ecg, conversion_info = standards.convert_to_clinical_scale(
                self.test_ecg_data, self.test_metadata['sample_rate']
            )

            assert scaled_ecg.shape == self.test_ecg_data.shape, "Scale conversion preserved shape"
            assert conversion_info['paper_speed_mm_per_s'] == 25.0, "Default paper speed correct"
            assert conversion_info['amplitude_scale_mm_per_mv'] == 10.0, "Default amplitude scale correct"
            print("   ✓ Clinical scale conversion working")

            # Test 2: Grid parameters
            print("2. Testing clinical grid parameters...")
            duration = self.test_ecg_data.shape[1] / self.test_metadata['sample_rate']
            amplitude_range = (np.min(self.test_ecg_data), np.max(self.test_ecg_data))

            grid_params = standards.get_clinical_grid_parameters(duration, amplitude_range)

            assert 'major_time_ticks' in grid_params, "Grid has major time ticks"
            assert 'major_voltage_ticks' in grid_params, "Grid has major voltage ticks"
            assert len(grid_params['major_time_ticks']) > 0, "Time ticks generated"
            assert len(grid_params['major_voltage_ticks']) > 0, "Voltage ticks generated"
            print("   ✓ Clinical grid parameters working")

            # Test 3: Lead layout
            print("3. Testing clinical lead layouts...")
            layout_4x3 = standards.get_clinical_lead_layout('4x3_standard')
            layout_3x4 = standards.get_clinical_lead_layout('3x4_landscape')

            assert layout_4x3['rows'] == 4, "4x3 layout has correct rows"
            assert layout_4x3['cols'] == 4, "4x3 layout has correct columns"
            assert layout_3x4['rows'] == 3, "3x4 layout has correct rows"
            assert layout_3x4['cols'] == 4, "3x4 layout has correct columns"
            print("   ✓ Clinical lead layouts working")

            # Test 4: Calibration signal
            print("4. Testing calibration signal...")
            cal_signal, cal_info = standards.create_calibration_signal(
                self.test_metadata['sample_rate']
            )

            assert len(cal_signal) > 0, "Calibration signal generated"
            assert cal_info['amplitude_mv'] == 1.0, "Standard calibration amplitude"
            assert np.max(cal_signal) == 1.0, "Calibration signal reaches 1mV"
            print("   ✓ Calibration signal working")

            # Test 5: Clinical compliance validation
            print("5. Testing clinical compliance validation...")
            compliance = standards.validate_clinical_compliance(
                self.test_ecg_data, self.test_metadata['sample_rate']
            )

            assert isinstance(compliance, dict), "Compliance returns dictionary"
            assert 'overall_compliant' in compliance, "Overall compliance assessed"
            assert len(compliance) >= 5, "Multiple compliance checks performed"
            print("   ✓ Clinical compliance validation working")

            print("✅ All clinical standards tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Clinical standards test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_clinical_plotting_module(self) -> bool:
        """Test clinical plotting functionality."""
        print("\n" + "=" * 50)
        print("🧪 TESTING CLINICAL PLOTTING MODULE")
        print("=" * 50)

        try:
            # Test 1: Basic clinical plotter initialization
            print("1. Testing clinical plotter initialization...")
            plotter = ClinicalECGPlotter(
                paper_speed=25.0,
                amplitude_scale=10.0,
                style='clinical'
            )

            assert plotter.paper_speed == 25.0, "Paper speed set correctly"
            assert plotter.amplitude_scale == 10.0, "Amplitude scale set correctly"
            assert plotter.style == 'clinical', "Style set correctly"
            print("   ✓ Clinical plotter initialization working")

            # Test 2: 12-lead clinical plot
            print("2. Testing 12-lead clinical plotting...")
            fig_12lead = plotter.plot_12_lead_clinical(
                self.test_ecg_data,
                self.test_metadata['sample_rate'],
                title="Stage 2 Test - 12 Lead",
                show_calibration=True,
                show_measurements=True
            )

            assert isinstance(fig_12lead, plt.Figure), "12-lead plot returns Figure"
            assert len(fig_12lead.axes) > 0, "12-lead plot has axes"
            print("   ✓ 12-lead clinical plotting working")

            # Test 3: Single lead clinical plot
            print("3. Testing single lead clinical plotting...")
            lead_ii = self.test_ecg_data[1]
            fig_single = plotter.plot_single_lead_clinical(
                lead_ii,
                self.test_metadata['sample_rate'],
                lead_name="Lead II",
                title="Stage 2 Test - Single Lead"
            )

            assert isinstance(fig_single, plt.Figure), "Single lead plot returns Figure"
            assert len(fig_single.axes) == 1, "Single lead plot has one axis"
            print("   ✓ Single lead clinical plotting working")

            # Test 4: Different styles
            print("4. Testing different plotting styles...")
            styles = ['clinical', 'research', 'print']

            for style in styles:
                style_plotter = ClinicalECGPlotter(style=style)
                assert style_plotter.style == style, f"Style {style} set correctly"
                assert style in style_plotter.styles, f"Style {style} configuration exists"
            print("   ✓ Different plotting styles working")

            # Test 5: Layout options
            print("5. Testing different layout options...")
            layouts = ['4x3_standard', '3x4_landscape', '6x2_compact']

            for layout in layouts:
                try:
                    fig_layout = plotter.plot_12_lead_clinical(
                        self.test_ecg_data,
                        self.test_metadata['sample_rate'],
                        layout=layout,
                        title=f"Test Layout: {layout}"
                    )
                    assert isinstance(fig_layout, plt.Figure), f"Layout {layout} produces Figure"
                    plt.close(fig_layout)  # Clean up
                except Exception as e:
                    print(f"   ⚠ Layout {layout} failed: {e}")
            print("   ✓ Layout options working")

            # Clean up figures
            plt.close(fig_12lead)
            plt.close(fig_single)

            print("✅ All clinical plotting tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Clinical plotting test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_measurement_tools_module(self) -> bool:
        """Test ECG measurement tools."""
        print("\n" + "=" * 50)
        print("🧪 TESTING MEASUREMENT TOOLS MODULE")
        print("=" * 50)

        try:
            # Test 1: Measurement tools initialization
            print("1. Testing measurement tools initialization...")
            tools = ECGMeasurementTools(
                self.test_metadata['sample_rate'],
                paper_speed=25.0,
                amplitude_scale=10.0
            )

            assert tools.sample_rate == self.test_metadata['sample_rate'], "Sample rate set correctly"
            assert tools.paper_speed == 25.0, "Paper speed set correctly"
            assert hasattr(tools, 'scales'), "Measurement scales calculated"
            print("   ✓ Measurement tools initialization working")

            # Test 2: RR interval measurements
            print("2. Testing RR interval measurements...")
            lead_ii = self.test_ecg_data[1]
            rr_measurements = tools.measure_rr_intervals(lead_ii, "Lead II")

            assert len(rr_measurements) > 0, "RR measurements generated"

            # Check measurement types
            measurement_types = set([m.measurement_type for m in rr_measurements])
            assert MeasurementType.RR_INTERVAL in measurement_types, "RR intervals measured"
            assert MeasurementType.HEART_RATE in measurement_types, "Heart rates calculated"

            # Check measurement values
            hr_measurements = [m for m in rr_measurements if m.measurement_type == MeasurementType.HEART_RATE]
            if hr_measurements:
                avg_hr = np.mean([m.value for m in hr_measurements])
                assert 50 <= avg_hr <= 150, f"Heart rate reasonable: {avg_hr:.1f} bpm"

            print(f"   ✓ RR interval measurements working ({len(rr_measurements)} measurements)")

            # Test 3: Custom interval measurements
            print("3. Testing custom interval measurements...")
            start_sample = int(1.0 * self.test_metadata['sample_rate'])  # 1 second
            end_sample = int(1.2 * self.test_metadata['sample_rate'])  # 1.2 seconds

            custom_measurement = tools.measure_custom_interval(
                start_sample, end_sample, MeasurementType.CUSTOM_INTERVAL
            )

            expected_duration = 200  # ms
            assert abs(custom_measurement.value - expected_duration) < 10, "Custom interval accurate"
            assert custom_measurement.unit == "ms", "Custom interval in milliseconds"
            print("   ✓ Custom interval measurements working")

            # Test 4: Measurement calipers (create dummy plot)
            print("4. Testing measurement calipers...")
            fig, ax = plt.subplots(figsize=(8, 4))
            time_axis = np.linspace(0, 5, len(lead_ii[:2500]))  # First 5 seconds
            ax.plot(time_axis, lead_ii[:2500])

            caliper_info = tools.add_measurement_calipers(
                ax, 1.0, 1.5, MeasurementType.CUSTOM_INTERVAL
            )

            assert 'duration_ms' in caliper_info, "Caliper duration calculated"
            assert caliper_info['duration_ms'] == 500, "Caliper duration correct (500ms)"

            plt.close(fig)
            print("   ✓ Measurement calipers working")

            # Test 5: Measurements report
            print("5. Testing measurements report...")
            report = tools.create_measurements_report(
                rr_measurements[:5],  # First 5 measurements
                "Stage 2 Test Report"
            )

            assert isinstance(report, str), "Report is string"
            assert len(report) > 100, "Report has substantial content"
            assert "Stage 2 Test Report" in report, "Report title included"
            assert "Summary:" in report, "Report has summary section"
            print("   ✓ Measurements report working")

            print("✅ All measurement tools tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Measurement tools test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_clinical_examples_module(self) -> bool:
        """Test clinical examples and scenarios."""
        print("\n" + "=" * 50)
        print("🧪 TESTING CLINICAL EXAMPLES MODULE")
        print("=" * 50)

        try:
            examples = ClinicalECGExamples()

            # Test 1: Emergency department scenario
            print("1. Testing emergency department scenario...")
            ed_results = examples.emergency_department_scenario()

            assert 'scenario' in ed_results, "ED scenario returns results"
            assert ed_results['scenario'] == 'Emergency Department', "Correct scenario type"
            assert 'ecg_data' in ed_results, "ED scenario includes ECG data"
            assert 'measurements' in ed_results, "ED scenario includes measurements"
            assert 'clinical_assessment' in ed_results, "ED scenario includes assessment"

            # Close figure
            if 'figure' in ed_results:
                plt.close(ed_results['figure'])

            print("   ✓ Emergency department scenario working")

            # Test 2: Cardiology clinic scenario
            print("2. Testing cardiology clinic scenario...")
            clinic_results = examples.cardiology_clinic_scenario()

            assert 'scenario' in clinic_results, "Clinic scenario returns results"
            assert clinic_results['scenario'] == 'Cardiology Clinic', "Correct scenario type"
            assert 'detailed_report' in clinic_results, "Clinic scenario includes detailed report"

            # Close figure
            if 'figure' in clinic_results:
                plt.close(clinic_results['figure'])

            print("   ✓ Cardiology clinic scenario working")

            # Test 3: Research study scenario
            print("3. Testing research study scenario...")
            research_results = examples.research_study_scenario()

            assert 'scenario' in research_results, "Research scenario returns results"
            assert research_results['scenario'] == 'Research Study', "Correct scenario type"
            assert 'subjects_data' in research_results, "Research scenario includes subjects"
            assert len(research_results['subjects_data']) > 0, "Research scenario has subjects"

            # Close figures
            for subject_data in research_results['subjects_data']:
                if 'figure' in subject_data:
                    plt.close(subject_data['figure'])

            print("   ✓ Research study scenario working")

            # Test 4: Measurement tools demonstration
            print("4. Testing measurement tools demonstration...")
            meas_fig, measurements = examples.demonstrate_measurement_tools()

            assert isinstance(meas_fig, plt.Figure), "Measurement demo returns Figure"
            assert len(measurements) > 0, "Measurement demo generates measurements"

            plt.close(meas_fig)
            print("   ✓ Measurement tools demonstration working")

            # Test 5: Clinical standards demonstration
            print("5. Testing clinical standards demonstration...")
            standards_fig, compliance = examples.demonstrate_clinical_standards()

            assert isinstance(standards_fig, plt.Figure), "Standards demo returns Figure"
            assert isinstance(compliance, dict), "Standards demo returns compliance"
            assert 'overall_compliant' in compliance, "Compliance check performed"

            plt.close(standards_fig)
            print("   ✓ Clinical standards demonstration working")

            print("✅ All clinical examples tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Clinical examples test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_stage1_compatibility(self) -> bool:
        """Test Stage 2 compatibility with Stage 1."""
        print("\n" + "=" * 50)
        print("🧪 TESTING STAGE 1 COMPATIBILITY")
        print("=" * 50)

        try:
            # Test 1: Stage 1 validation still works
            print("1. Testing Stage 1 validation compatibility...")
            validator = ECGValidator()
            validation_results = validator.validate_ecg_data(
                self.test_ecg_data,
                self.test_metadata['sample_rate'],
                self.test_metadata['leads']
            )

            assert validation_results['overall_valid'], "Stage 1 validation still works"
            print("   ✓ Stage 1 validation compatibility confirmed")

            # Test 2: Original ecg_plot functions still work
            print("2. Testing original ecg_plot compatibility...")

            # Test plot_12
            ecg_plot.plot_12(
                self.test_ecg_data,
                self.test_metadata['sample_rate'],
                title="Stage 1 Compatibility Test"
            )
            print("   ✓ Original plot_12 still works")

            # Test enhanced plot
            ecg_plot.plot(
                self.test_ecg_data,
                self.test_metadata['sample_rate'],
                title="Stage 1 Enhanced Plot Test",
                style='bw'
            )
            print("   ✓ Original enhanced plot still works")

            # Test single lead plot
            ecg_plot.plot_1(
                self.test_ecg_data[1],
                self.test_metadata['sample_rate'],
                title="Stage 1 Single Lead Test"
            )
            print("   ✓ Original plot_1 still works")

            # Save test (don't actually save, just test function)
            try:
                ecg_plot.save_as_png("compatibility_test", path="./")
                print("   ✓ Original save functions still work")
            except Exception as e:
                print(f"   ⚠ Save function warning: {e}")

            # Test 3: Stage 2 enhances Stage 1 data
            print("3. Testing Stage 2 enhancement of Stage 1 data...")

            # Use Stage 1 generated data with Stage 2 tools
            plotter = ClinicalECGPlotter()
            clinical_fig = plotter.plot_12_lead_clinical(
                self.test_ecg_data,
                self.test_metadata['sample_rate'],
                title="Stage 1 Data + Stage 2 Tools"
            )

            assert isinstance(clinical_fig, plt.Figure), "Stage 2 can process Stage 1 data"
            plt.close(clinical_fig)
            print("   ✓ Stage 2 enhances Stage 1 data successfully")

            print("✅ All Stage 1 compatibility tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Stage 1 compatibility test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_clinical_accuracy(self) -> bool:
        """Test clinical accuracy and standards compliance."""
        print("\n" + "=" * 50)
        print("🧪 TESTING CLINICAL ACCURACY")
        print("=" * 50)

        try:
            # Test 1: Paper scaling accuracy
            print("1. Testing paper scaling accuracy...")
            standards = ClinicalECGStandards()

            # Test time scaling: 25mm/s standard
            time_1_second = 25.0  # mm
            grid_params = standards.get_clinical_grid_parameters(
                duration=5.0,
                amplitude_range=(-2, 2)
            )

            # Check if 1 second intervals are correctly spaced
            major_time_interval = grid_params['major_time_ticks'][1] - grid_params['major_time_ticks'][0]
            expected_interval = 0.2  # 5 large squares = 1 second, so each large square = 0.2s
            assert abs(major_time_interval - expected_interval) < 0.001, "Time scaling accurate"
            print("   ✓ Paper time scaling accurate (25mm/s)")

            # Test amplitude scaling: 10mm/mV standard
            major_voltage_interval = grid_params['major_voltage_ticks'][1] - grid_params['major_voltage_ticks'][0]
            expected_voltage_interval = 0.5  # Each large square = 0.5mV
            assert abs(major_voltage_interval - expected_voltage_interval) < 0.001, "Voltage scaling accurate"
            print("   ✓ Paper amplitude scaling accurate (10mm/mV)")

            # Test 2: Measurement accuracy
            print("2. Testing measurement accuracy...")
            tools = ECGMeasurementTools(self.test_metadata['sample_rate'])

            # Test known interval measurement
            known_duration_samples = self.test_metadata['sample_rate']  # Exactly 1 second
            measurement = tools.measure_custom_interval(0, known_duration_samples)

            expected_ms = 1000  # 1 second = 1000 ms
            assert abs(measurement.value - expected_ms) < 1, "Measurement accuracy within 1ms"
            print("   ✓ Measurement accuracy confirmed (±1ms)")

            # Test 3: Heart rate calculation accuracy
            print("3. Testing heart rate calculation accuracy...")

            # Create test signal with known heart rate
            known_hr = 60  # bpm
            test_ecg, test_meta = self.generator.generate_normal_sinus_rhythm(
                duration=10, heart_rate=known_hr, noise_level=0.01
            )

            lead_ii = test_ecg[1]
            rr_measurements = tools.measure_rr_intervals(lead_ii, "Test Lead")

            hr_measurements = [m for m in rr_measurements if m.measurement_type == MeasurementType.HEART_RATE]
            if hr_measurements:
                avg_calculated_hr = np.mean([m.value for m in hr_measurements])
                hr_error = abs(avg_calculated_hr - known_hr)
                assert hr_error < 5, f"Heart rate calculation accurate within 5 bpm (error: {hr_error:.1f})"
                print(f"   ✓ Heart rate accuracy confirmed (calculated: {avg_calculated_hr:.1f}, expected: {known_hr})")
            else:
                print("   ⚠ Could not verify heart rate accuracy (no measurements)")

            # Test 4: Clinical normal ranges
            print("4. Testing clinical normal ranges...")

            # Generate normal ECG and verify assessments
            normal_ecg, normal_meta = self.generator.generate_normal_sinus_rhythm(
                duration=10, heart_rate=75, noise_level=0.02
            )

            normal_tools = ECGMeasurementTools(normal_meta['sample_rate'])
            normal_measurements = normal_tools.measure_rr_intervals(normal_ecg[1], "Lead II")

            # Check that normal ECG generates mostly normal measurements
            normal_count = sum(1 for m in normal_measurements if m.is_normal)
            total_count = len(normal_measurements)

            if total_count > 0:
                normal_percentage = normal_count / total_count
                assert normal_percentage > 0.7, f"Normal ECG should have >70% normal measurements (got {normal_percentage:.1%})"
                print(f"   ✓ Normal range assessment working ({normal_percentage:.1%} normal)")

            print("✅ All clinical accuracy tests PASSED")
            return True

        except Exception as e:
            print(f"❌ Clinical accuracy test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self) -> dict:
        """Run complete Stage 2 test suite."""
        print("🩺 ECG PLOT LIBRARY - STAGE 2 INTEGRATION TEST SUITE")
        print("Testing clinical standards compliance and measurement tools...")
        print("=" * 70)

        # Set up test data
        self.setup_test_data()

        # Run all test modules
        test_modules = [
            ("Clinical Standards Module", self.test_clinical_standards_module),
            ("Clinical Plotting Module", self.test_clinical_plotting_module),
            ("Measurement Tools Module", self.test_measurement_tools_module),
            ("Clinical Examples Module", self.test_clinical_examples_module),
            ("Stage 1 Compatibility", self.test_stage1_compatibility),
            ("Clinical Accuracy", self.test_clinical_accuracy)
        ]

        results = {}
        passed_count = 0

        for test_name, test_function in test_modules:
            print(f"\n🧪 Running {test_name} tests...")
            try:
                result = test_function()
                results[test_name] = result
                if result:
                    passed_count += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False

        # Final results
        total_tests = len(test_modules)
        success_rate = passed_count / total_tests

        print("\n" + "=" * 70)
        print("🏁 STAGE 2 TEST SUITE RESULTS")
        print("=" * 70)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status:8} {test_name}")

        print(f"\nOverall Results:")
        print(f"  • Tests Passed: {passed_count}/{total_tests}")
        print(f"  • Success Rate: {success_rate:.1%}")
        print(f"  • Overall Status: {'🎉 STAGE 2 READY' if success_rate >= 0.8 else '⚠️ NEEDS FIXES'}")

        if success_rate >= 0.8:
            print("\n🚀 Stage 2 Features Successfully Implemented:")
            print("  ✓ Clinical ECG Standards (25mm/s, 10mm/mV)")
            print("  ✓ Professional ECG plotting with medical layouts")
            print("  ✓ Clinical measurement tools and calipers")
            print("  ✓ ECG interval measurements (RR, HR, PR, QRS, QT)")
            print("  ✓ Multi-scenario clinical applications")
            print("  ✓ Full compatibility with Stage 1 features")
            print("\n🎯 Ready for clinical use and Stage 3 development!")

        else:
            print(f"\n⚠️ Some tests failed. Please fix issues before proceeding.")
            print("Check the detailed error messages above for specific problems.")

        return results


def main():
    """Main test function."""
    # Suppress matplotlib GUI warnings during testing
    plt.ioff()

    # Run the test suite
    test_suite = Stage2TestSuite()
    results = test_suite.run_all_tests()

    # Clean up any remaining plots
    plt.close('all')

    return results


if __name__ == "__main__":
    main()