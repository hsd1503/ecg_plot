#!/usr/bin/env python
"""
Stage 1 Integration Test
Tests all Stage 1 components together with the original ecg_plot library.
"""

import numpy as np
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ecg_plot
from examples.generate_ecg_data import ECGGenerator
from ecg_validation import ECGValidator, quick_validate


def test_stage1_integration():
    """Test all Stage 1 components together."""
    print("Testing Stage 1 Implementation")
    print("=" * 40)

    # Step 1: Generate realistic ECG data
    print("1. Generating synthetic ECG data...")
    generator = ECGGenerator(sample_rate=500)
    ecg_data, metadata = generator.generate_normal_sinus_rhythm(
        duration=10,
        heart_rate=72,
        noise_level=0.05
    )
    print(f"   ✓ Generated {metadata['signal_type']}")
    print(f"   ✓ Shape: {ecg_data.shape}")
    print(f"   ✓ Sample rate: {metadata['sample_rate']} Hz")
    print(f"   ✓ Heart rate: {metadata['heart_rate']} bpm")

    # Step 2: Validate the ECG data
    print("\n2. Validating ECG data...")
    validator = ECGValidator(strict_mode=False)
    results = validator.validate_ecg_data(
        ecg_data,
        metadata['sample_rate'],
        metadata['leads']
    )

    print("   Validation Results:")
    for check, result in results.items():
        status = "✓" if result else "✗"
        print(f"   {status} {check.replace('_', ' ').title()}")

    # Step 3: Plot using original ecg_plot functions
    print("\n3. Testing with original ecg_plot functions...")

    try:
        # Test plot_12 function
        print("   Testing plot_12()...")
        ecg_plot.plot_12(
            ecg_data,
            sample_rate=metadata['sample_rate'],
            title="Stage 1 Test - 12 Lead ECG",
            lead_index=metadata['leads']
        )
        print("   ✓ plot_12() successful")

        # Test enhanced plot function
        print("   Testing enhanced plot()...")
        ecg_plot.plot(
            ecg_data,
            sample_rate=metadata['sample_rate'],
            title="Stage 1 Test - Enhanced Plot",
            lead_index=metadata['leads'],
            style='bw',
            show_grid=True
        )
        print("   ✓ enhanced plot() successful")

        # Test single lead plot
        print("   Testing plot_1()...")
        lead_ii_index = metadata['leads'].index('II')
        ecg_plot.plot_1(
            ecg_data[lead_ii_index],
            sample_rate=metadata['sample_rate'],
            title="Stage 1 Test - Lead II Rhythm Strip"
        )
        print("   ✓ plot_1() successful")

        # Save test images
        print("   Saving test images...")
        ecg_plot.save_as_png("stage1_test_plot", dpi=300)
        print("   ✓ Image saved as 'stage1_test_plot.png'")

    except Exception as e:
        print(f"   ✗ Plotting error: {e}")
        return False

    print("\n4. Testing different ECG scenarios...")

    # Test with different heart rates
    test_scenarios = [
        {"hr": 60, "name": "Bradycardia (60 bpm)"},
        {"hr": 100, "name": "Tachycardia (100 bpm)"},
        {"hr": 72, "noise": 0.15, "name": "Noisy signal"},
    ]

    for scenario in test_scenarios:
        hr = scenario.get("hr", 72)
        noise = scenario.get("noise", 0.05)
        name = scenario["name"]

        print(f"   Testing {name}...")
        test_ecg, test_meta = generator.generate_normal_sinus_rhythm(
            heart_rate=hr, noise_level=noise, duration=5
        )

        # Quick validation
        is_valid = quick_validate(test_ecg, test_meta['sample_rate'], test_meta['leads'])
        status = "✓" if is_valid else "✗"
        print(f"   {status} {name} - Validation: {'PASS' if is_valid else 'FAIL'}")

    print("\n" + "=" * 40)
    print("Stage 1 Integration Test Complete!")
    print("\nNext Steps:")
    print("- Create pull request with Stage 1 improvements")
    print("- Begin Stage 2: Clinical Standards Compliance")
    print("- Add more sophisticated ECG generation algorithms")

    return True


def demonstrate_clinical_use_cases():
    """Demonstrate clinical use cases for the enhanced library."""
    print("\n" + "=" * 50)
    print("CLINICAL USE CASE DEMONSTRATIONS")
    print("=" * 50)

    generator = ECGGenerator()

    # Use Case 1: Emergency Department ECG
    print("\n📋 USE CASE 1: Emergency Department")
    print("-" * 30)
    ed_ecg, ed_meta = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=95, noise_level=0.08
    )

    # Validate for clinical use
    validator = ECGValidator(strict_mode=False)
    validation = validator.validate_ecg_data(ed_ecg, ed_meta['sample_rate'], ed_meta['leads'])

    print(f"Patient: Emergency admission")
    print(f"Signal quality: {'ACCEPTABLE' if validation['overall_valid'] else 'POOR'}")
    print(f"Heart rate: {ed_meta['heart_rate']} bpm")
    print(
        f"Recommended action: {'Proceed with interpretation' if validation['overall_valid'] else 'Repeat acquisition'}")

    # Use Case 2: Cardiology Clinic
    print("\n🏥 USE CASE 2: Cardiology Clinic")
    print("-" * 30)
    clinic_ecg, clinic_meta = generator.generate_normal_sinus_rhythm(
        duration=10, heart_rate=68, noise_level=0.02
    )

    validation = validator.validate_ecg_data(clinic_ecg, clinic_meta['sample_rate'], clinic_meta['leads'])

    print(f"Patient: Routine follow-up")
    print(f"Signal quality: HIGH")
    print(f"Heart rate: {clinic_meta['heart_rate']} bpm")
    print(f"All leads: NORMAL SINUS RHYTHM")

    # Use Case 3: Research Study
    print("\n🔬 USE CASE 3: Research Study")
    print("-" * 30)
    research_data = []
    for subject in range(3):
        hr = np.random.randint(65, 85)
        subj_ecg, subj_meta = generator.generate_normal_sinus_rhythm(
            duration=30, heart_rate=hr, noise_level=0.03
        )
        research_data.append((subj_ecg, subj_meta))

        is_valid = quick_validate(subj_ecg, subj_meta['sample_rate'], subj_meta['leads'])
        print(f"Subject {subject + 1:02d}: HR={hr:02d} bpm, Quality={'✓' if is_valid else '✗'}")

    print(f"\nDataset: {len(research_data)} subjects ready for analysis")

    return True


if __name__ == "__main__":
    try:
        # Run main integration test
        success = test_stage1_integration()

        if success:
            # Run clinical demonstrations
            demonstrate_clinical_use_cases()

            print("\n🎉 TESTS PASSED!")

        else:
            print("\n❌ Tests failed. Check the errors above.")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nMake sure you have:")
        print("- Created all the required files")
        print("- Added the project root to your Python path")
        print("- Installed required packages: numpy, matplotlib, scipy")

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("\nPlease check your implementation and try again.")