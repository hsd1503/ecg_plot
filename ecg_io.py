#!/usr/bin/env python
"""
ECG File Format I/O Module
Support for standard ECG file formats (WFDB, EDF, CSV, JSON).
"""

import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import struct


@dataclass
class ECGMetadata:
    """ECG recording metadata."""
    patient_id: str
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

    sample_rate: int = 500
    duration: float = 10.0
    n_leads: int = 12
    lead_names: List[str] = None

    device: Optional[str] = None
    technician: Optional[str] = None
    comments: Optional[str] = None

    def __post_init__(self):
        if self.lead_names is None:
            self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                               'V1', 'V2', 'V3', 'V4', 'V5', 'V6'][:self.n_leads]
        if self.date is None:
            self.date = datetime.now().strftime('%Y-%m-%d')
        if self.time is None:
            self.time = datetime.now().strftime('%H:%M:%S')


class ECGFileReader:
    """Read ECG data from various file formats."""

    @staticmethod
    def read_csv(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Read ECG from CSV file.

        CSV Format:
        Row 1: Header with lead names
        Rows 2+: Sample values for each lead

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (ecg_data, metadata)
        """
        with open(filepath, 'r') as f:
            reader = csv.reader(f)

            # Read header
            header = next(reader)

            # Check for metadata row
            first_data_row = next(reader)
            if first_data_row[0].startswith('#'):
                # Metadata present
                sample_rate = int(first_data_row[1]) if len(first_data_row) > 1 else 500
                first_data_row = next(reader)
            else:
                sample_rate = 500

            # Read all data
            data = [first_data_row]
            for row in reader:
                if row and not row[0].startswith('#'):
                    data.append(row)

            # Convert to numpy array
            ecg_data = np.array(data, dtype=float).T

            # Create metadata
            metadata = ECGMetadata(
                patient_id='CSV_Import',
                sample_rate=sample_rate,
                duration=ecg_data.shape[1] / sample_rate,
                n_leads=ecg_data.shape[0],
                lead_names=header[:ecg_data.shape[0]]
            )

        return ecg_data, metadata

    @staticmethod
    def read_json(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Read ECG from JSON file.

        JSON Format:
        {
            "metadata": {...},
            "data": [[lead1_samples], [lead2_samples], ...]
        }

        Args:
            filepath: Path to JSON file

        Returns:
            Tuple of (ecg_data, metadata)
        """
        with open(filepath, 'r') as f:
            ecg_json = json.load(f)

        # Extract data
        ecg_data = np.array(ecg_json['data'])

        # Extract metadata
        meta_dict = ecg_json.get('metadata', {})

        metadata = ECGMetadata(
            patient_id=meta_dict.get('patient_id', 'JSON_Import'),
            patient_name=meta_dict.get('patient_name'),
            age=meta_dict.get('age'),
            sex=meta_dict.get('sex'),
            date=meta_dict.get('date'),
            time=meta_dict.get('time'),
            sample_rate=meta_dict.get('sample_rate', 500),
            duration=meta_dict.get('duration', ecg_data.shape[1] / meta_dict.get('sample_rate', 500)),
            n_leads=ecg_data.shape[0],
            lead_names=meta_dict.get('lead_names'),
            device=meta_dict.get('device'),
            technician=meta_dict.get('technician'),
            comments=meta_dict.get('comments')
        )

        return ecg_data, metadata

    @staticmethod
    def read_wfdb(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Read ECG from WFDB format (PhysioNet).

        Note: This is a simplified implementation.
        For production use, install wfdb package: pip install wfdb

        Args:
            filepath: Path to WFDB file (without extension)

        Returns:
            Tuple of (ecg_data, metadata)
        """
        try:
            import wfdb

            # Read record
            record = wfdb.rdrecord(filepath)

            # Extract data
            ecg_data = record.p_signal.T  # Transpose to (leads, samples)

            # Extract metadata
            metadata = ECGMetadata(
                patient_id=filepath.split('/')[-1],
                sample_rate=record.fs,
                duration=record.sig_len / record.fs,
                n_leads=record.n_sig,
                lead_names=record.sig_name,
                comments=record.comments[0] if record.comments else None
            )

            return ecg_data, metadata

        except ImportError:
            raise ImportError("WFDB package not installed. Install with: pip install wfdb")

    @staticmethod
    def read_edf(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Read ECG from EDF format (European Data Format).

        Note: This is a simplified implementation.
        For production use, install pyedflib: pip install pyedflib

        Args:
            filepath: Path to EDF file

        Returns:
            Tuple of (ecg_data, metadata)
        """
        try:
            import pyedflib

            # Open EDF file
            f = pyedflib.EdfReader(filepath)

            # Read signals
            n_signals = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sample_rate = f.getSampleFrequency(0)

            # Read all signals
            signals = []
            for i in range(n_signals):
                signals.append(f.readSignal(i))

            ecg_data = np.array(signals)

            # Read patient info
            header = f.getHeader()
            patient_info = f.getPatientAdditional()

            metadata = ECGMetadata(
                patient_id=header.get('patientcode', 'EDF_Import'),
                sample_rate=int(sample_rate),
                duration=len(signals[0]) / sample_rate,
                n_leads=n_signals,
                lead_names=signal_labels,
                date=header.get('startdate').strftime('%Y-%m-%d') if header.get('startdate') else None
            )

            f.close()

            return ecg_data, metadata

        except ImportError:
            raise ImportError("pyedflib package not installed. Install with: pip install pyedflib")

    @staticmethod
    def read_numpy(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Read ECG from numpy format (.npy or .npz).

        Args:
            filepath: Path to numpy file

        Returns:
            Tuple of (ecg_data, metadata)
        """
        if filepath.endswith('.npz'):
            # Load npz with metadata
            data = np.load(filepath, allow_pickle=True)
            ecg_data = data['ecg']

            if 'metadata' in data:
                meta_dict = data['metadata'].item()
                metadata = ECGMetadata(**meta_dict)
            else:
                metadata = ECGMetadata(
                    patient_id='NPZ_Import',
                    sample_rate=data.get('sample_rate', 500),
                    duration=ecg_data.shape[1] / data.get('sample_rate', 500),
                    n_leads=ecg_data.shape[0]
                )
        else:
            # Load npy (data only)
            ecg_data = np.load(filepath)
            metadata = ECGMetadata(
                patient_id='NPY_Import',
                duration=ecg_data.shape[1] / 500,
                n_leads=ecg_data.shape[0]
            )

        return ecg_data, metadata

    @staticmethod
    def auto_detect_format(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
        """
        Automatically detect and read ECG file format.

        Args:
            filepath: Path to ECG file

        Returns:
            Tuple of (ecg_data, metadata)
        """
        filepath_lower = filepath.lower()

        if filepath_lower.endswith('.csv'):
            return ECGFileReader.read_csv(filepath)
        elif filepath_lower.endswith('.json'):
            return ECGFileReader.read_json(filepath)
        elif filepath_lower.endswith('.npy') or filepath_lower.endswith('.npz'):
            return ECGFileReader.read_numpy(filepath)
        elif filepath_lower.endswith('.edf'):
            return ECGFileReader.read_edf(filepath)
        else:
            # Try WFDB format (no extension)
            try:
                return ECGFileReader.read_wfdb(filepath)
            except:
                raise ValueError(f"Unknown or unsupported file format: {filepath}")


class ECGFileWriter:
    """Write ECG data to various file formats."""

    @staticmethod
    def write_csv(filepath: str, ecg_data: np.ndarray, metadata: ECGMetadata):
        """
        Write ECG to CSV file.

        Args:
            filepath: Output file path
            ecg_data: ECG signal array (leads, samples)
            metadata: ECG metadata
        """
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(metadata.lead_names)

            # Write metadata row
            writer.writerow([f'# Sample Rate: {metadata.sample_rate} Hz'])

            # Write data (transpose to have samples as rows)
            for sample_idx in range(ecg_data.shape[1]):
                row = ecg_data[:, sample_idx].tolist()
                writer.writerow(row)

    @staticmethod
    def write_json(filepath: str, ecg_data: np.ndarray, metadata: ECGMetadata):
        """
        Write ECG to JSON file.

        Args:
            filepath: Output file path
            ecg_data: ECG signal array (leads, samples)
            metadata: ECG metadata
        """
        ecg_json = {
            'metadata': asdict(metadata),
            'data': ecg_data.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(ecg_json, f, indent=2)

    @staticmethod
    def write_numpy(filepath: str, ecg_data: np.ndarray, metadata: ECGMetadata,
                    compressed: bool = True):
        """
        Write ECG to numpy format.

        Args:
            filepath: Output file path
            ecg_data: ECG signal array (leads, samples)
            metadata: ECG metadata
            compressed: Use npz format (compressed)
        """
        if compressed:
            np.savez_compressed(
                filepath,
                ecg=ecg_data,
                metadata=asdict(metadata),
                sample_rate=metadata.sample_rate
            )
        else:
            np.save(filepath, ecg_data)

    @staticmethod
    def write_wfdb(filepath: str, ecg_data: np.ndarray, metadata: ECGMetadata):
        """
        Write ECG to WFDB format.

        Requires wfdb package: pip install wfdb
        """
        try:
            import wfdb

            # Transpose data for WFDB format
            p_signal = ecg_data.T

            # Write record
            wfdb.wrsamp(
                record_name=filepath,
                fs=metadata.sample_rate,
                units=['mV'] * metadata.n_leads,
                sig_name=metadata.lead_names,
                p_signal=p_signal,
                comments=[metadata.comments] if metadata.comments else None
            )

        except ImportError:
            raise ImportError("WFDB package not installed. Install with: pip install wfdb")


# Convenience functions
def load_ecg(filepath: str) -> Tuple[np.ndarray, ECGMetadata]:
    """Load ECG from file (auto-detect format)."""
    reader = ECGFileReader()
    return reader.auto_detect_format(filepath)


def save_ecg(filepath: str, ecg_data: np.ndarray, metadata: ECGMetadata,
             format: Optional[str] = None):
    """
    Save ECG to file.

    Args:
        filepath: Output file path
        ecg_data: ECG signal array
        metadata: ECG metadata
        format: File format ('csv', 'json', 'numpy', 'wfdb'). Auto-detect if None.
    """
    if format is None:
        # Auto-detect from extension
        filepath_lower = filepath.lower()
        if filepath_lower.endswith('.csv'):
            format = 'csv'
        elif filepath_lower.endswith('.json'):
            format = 'json'
        elif filepath_lower.endswith('.npy') or filepath_lower.endswith('.npz'):
            format = 'numpy'
        else:
            format = 'json'  # Default

    writer = ECGFileWriter()

    if format == 'csv':
        writer.write_csv(filepath, ecg_data, metadata)
    elif format == 'json':
        writer.write_json(filepath, ecg_data, metadata)
    elif format == 'numpy':
        writer.write_numpy(filepath, ecg_data, metadata)
    elif format == 'wfdb':
        writer.write_wfdb(filepath, ecg_data, metadata)
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Test file I/O
    from examples.generate_ecg_data import ECGGenerator

    print("Testing ECG File I/O...")

    # Generate test ECG
    gen = ECGGenerator()
    ecg_data, gen_metadata = gen.generate_normal_sinus_rhythm(duration=10)

    # Create metadata
    metadata = ECGMetadata(
        patient_id='TEST001',
        patient_name='Test Patient',
        age=45,
        sex='M',
        sample_rate=gen_metadata['sample_rate'],
        duration=10.0,
        n_leads=12,
        lead_names=gen_metadata['leads']
    )

    # Test CSV
    print("\n1. Testing CSV format...")
    save_ecg('test_ecg.csv', ecg_data, metadata)
    loaded_ecg, loaded_meta = load_ecg('test_ecg.csv')
    print(f"   ✓ CSV: Saved and loaded {loaded_ecg.shape}")

    # Test JSON
    print("\n2. Testing JSON format...")
    save_ecg('test_ecg.json', ecg_data, metadata)
    loaded_ecg, loaded_meta = load_ecg('test_ecg.json')
    print(f"   ✓ JSON: Saved and loaded {loaded_ecg.shape}")

    # Test NumPy
    print("\n3. Testing NumPy format...")
    save_ecg('test_ecg.npz', ecg_data, metadata)
    loaded_ecg, loaded_meta = load_ecg('test_ecg.npz')
    print(f"   NumPy: Saved and loaded {loaded_ecg.shape}")

    print("\nFile I/O module working!")