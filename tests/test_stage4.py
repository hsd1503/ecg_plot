#!/usr/bin/env python
"""
Stage 4 Integration Test Suite
File I/O & Clinical Reporting
"""

import numpy as np
import sys, os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.generate_ecg_data import ECGGenerator
from ecg_io import ECGMetadata, ECGFileReader, ECGFileWriter, load_ecg, save_ecg
from clinical_report import ClinicalReportGenerator, generate_quick_report


def test_stage4():
    """Complete Stage 4 test suite."""
    print("🩺 STAGE 4 INTEGRATION TEST SUITE")
    print("=" * 60)

    # Generate test ECG
    gen = ECGGenerator()
    ecg_data, gen_meta = gen.generate_normal_sinus_rhythm(duration=10, heart_rate=75)

    # Create metadata
    metadata = ECGMetadata(
        patient_id='TEST001',
        patient_name='Test Patient',
        age=45,
        sex='M',
        sample_rate=gen_meta['sample_rate'],
        duration=10.0,
        n_leads=12,
        lead_names=gen_meta['leads']
    )

    results = {}

    # Test 1: CSV I/O
    print("\n1. Testing CSV File I/O...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name

        save_ecg(csv_path, ecg_data, metadata, format='csv')
        loaded_ecg, loaded_meta = load_ecg(csv_path)

        assert loaded_ecg.shape == ecg_data.shape, "CSV: Shape preserved"
        assert loaded_meta.sample_rate == metadata.sample_rate, "CSV: Sample rate preserved"

        results['csv_io'] = True
        print(f" CSV I/O: {loaded_ecg.shape}")

        os.remove(csv_path)
    except Exception as e:
        print(f" CSV I/O failed: {e}")
        results['csv_io'] = False

    # Test 2: JSON I/O
    print("\n2. Testing JSON File I/O...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name

        save_ecg(json_path, ecg_data, metadata, format='json')
        loaded_ecg, loaded_meta = load_ecg(json_path)

        assert loaded_ecg.shape == ecg_data.shape, "JSON: Shape preserved"
        assert loaded_meta.patient_id == metadata.patient_id, "JSON: Metadata preserved"

        results['json_io'] = True
        print(f" JSON I/O: {loaded_ecg.shape}")

        os.remove(json_path)
    except Exception as e:
        print(f" JSON I/O failed: {e}")
        results['json_io'] = False

    # Test 3: NumPy I/O
    print("\n3. Testing NumPy File I/O...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            npz_path = f.name

        save_ecg(npz_path, ecg_data, metadata, format='numpy')
        loaded_ecg, loaded_meta = load_ecg(npz_path)

        assert loaded_ecg.shape == ecg_data.shape, "NumPy: Shape preserved"
        assert np.allclose(loaded_ecg, ecg_data, rtol=1e-5), "NumPy: Data preserved"

        results['numpy_io'] = True
        print(f" NumPy I/O: {loaded_ecg.shape}")

        os.remove(npz_path)
    except Exception as e:
        print(f" NumPy I/O failed: {e}")
        results['numpy_io'] = False

    # Test 4: Clinical Report Generation
    print("\n4. Testing Clinical Report Generation...")
    try:
        generator = ClinicalReportGenerator(metadata.sample_rate)
        findings = generator.analyze_ecg(ecg_data, metadata)

        assert findings.heart_rate > 0, "Heart rate calculated"
        assert findings.rhythm is not None, "Rhythm detected"
        assert len(findings.findings) > 0, "Findings generated"

        results['report_generation'] = True
        print(f" Clinical Report: HR={findings.heart_rate:.0f} bpm, Rhythm={findings.rhythm}")
    except Exception as e:
        print(f" Report generation failed: {e}")
        results['report_generation'] = False

    # Test 5: Text Report
    print("\n5. Testing Text Report...")
    try:
        text_report = generator.generate_text_report(metadata, findings)

        assert len(text_report) > 100, "Report has content"
        assert "PATIENT INFORMATION" in text_report, "Report has patient section"
        assert "MEASUREMENTS" in text_report, "Report has measurements section"
        assert "INTERPRETATION" in text_report, "Report has interpretation section"

        results['text_report'] = True
        print("  Text Report: Generated successfully")
    except Exception as e:
        print(f" Text report failed: {e}")
        results['text_report'] = False

    # Test 6: PDF Report
    print("\n6. Testing PDF Report...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_path = f.name

        generator.generate_pdf_report(pdf_path, ecg_data, metadata, findings)

        assert os.path.exists(pdf_path), "PDF file created"
        assert os.path.getsize(pdf_path) > 1000, "PDF has content"

        results['pdf_report'] = True
        print("  PDF Report: Generated successfully")

        os.remove(pdf_path)
    except Exception as e:
        print(f" PDF report failed: {e}")
        results['pdf_report'] = False

    # Summary
    print("\n" + "=" * 60)
    print("STAGE 4 TEST RESULTS")
    print("=" * 60)

    test_names = {
        'csv_io': 'CSV File I/O',
        'json_io': 'JSON File I/O',
        'numpy_io': 'NumPy File I/O',
        'report_generation': 'Clinical Report Generation',
        'text_report': 'Text Report',
        'pdf_report': 'PDF Report'
    }

    for test_key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        test_name = test_names.get(test_key, test_key)
        print(f"{status:8} {test_name}")

    success_rate = sum(results.values()) / len(results)
    print(f"\nOverall: {sum(results.values())}/{len(results)} tests passed ({success_rate:.0%})")

    if success_rate == 1.0:
        print("\nSTAGE 4 IMPLEMENTATION COMPLETE!")
        print("File I/O working (CSV, JSON, NumPy)")
        print("Clinical report generation working")
        print("Automated ECG interpretation working")
        print("PDF reports with plots working")
        print("\nSystem complete and ready for production!")
    elif success_rate >= 0.8:
        print("\nStage 4 mostly working - minor issues to fix")
    else:
        print("\nStage 4 needs attention - check failed tests")

    return results


if __name__ == "__main__":
    test_stage4()