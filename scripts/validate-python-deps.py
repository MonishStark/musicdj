#!/usr/bin/env python3
"""
Python Dependencies Validation Script

This script validates that all required Python dependencies are properly installed
and can be imported successfully. It also performs basic functionality tests.
"""

import sys
import importlib
import subprocess
import json
from pathlib import Path

# Required dependencies with their expected versions
CORE_DEPENDENCIES = {
    'librosa': '0.11.0',
    'numpy': '1.23.5',
    'scipy': '1.15.2',
    'soundfile': '0.13.1',
    'pydub': '0.25.1',
    'madmom': '0.16.1',
    'spleeter': '2.4.0',
    'tensorflow': '2.9.3',
    'scikit-learn': '1.6.1',
    'joblib': '1.4.2',
    'pandas': '1.5.3',
    'requests': '2.32.3'
}

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version compatible")
    return True

def check_dependency(package_name, expected_version=None):
    """Check if a dependency can be imported and optionally verify version."""
    try:
        module = importlib.import_module(package_name)
        
        if expected_version:
            # Try to get version
            version = None
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            
            if version:
                if version == expected_version:
                    print(f"✅ {package_name} {version} (matches expected)")
                else:
                    print(f"⚠️  {package_name} {version} (expected {expected_version})")
            else:
                print(f"✅ {package_name} (imported, version unknown)")
        else:
            print(f"✅ {package_name} (imported)")
        
        return True
        
    except ImportError as e:
        print(f"❌ {package_name} - ImportError: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name} - Error: {e}")
        return False

def test_audio_processing():
    """Test basic audio processing functionality."""
    print("\n🔊 Testing audio processing capabilities...")
    
    try:
        import numpy as np
        import librosa
        
        # Create a simple test signal
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        test_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test tempo detection
        tempo, beats = librosa.beat.beat_track(y=test_signal, sr=sr)
        print(f"✅ Librosa tempo detection: {tempo:.1f} BPM")
        
        # Test spectral features
        mfccs = librosa.feature.mfcc(y=test_signal, sr=sr, n_mfcc=13)
        print(f"✅ MFCC extraction: shape {mfccs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow functionality."""
    print("\n🧠 Testing TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Check if TensorFlow can create tensors
        test_tensor = tf.constant([1.0, 2.0, 3.0])
        print(f"✅ TensorFlow tensor creation: {test_tensor.numpy()}")
        
        # Check GPU availability (optional)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU devices available: {len(gpus)}")
        else:
            print("ℹ️  No GPU devices found (CPU-only mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def check_security_vulnerabilities():
    """Check for known security vulnerabilities in dependencies."""
    print("\n🔒 Checking for security vulnerabilities...")
    
    try:
        # Try to run safety check if available
        result = subprocess.run(
            [sys.executable, '-m', 'safety', 'check', '--json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            vulnerabilities = json.loads(result.stdout)
            if vulnerabilities:
                print(f"❌ Found {len(vulnerabilities)} security vulnerabilities")
                for vuln in vulnerabilities[:3]:  # Show first 3
                    print(f"   - {vuln.get('package', 'Unknown')}: {vuln.get('advisory', 'No details')}")
                return False
            else:
                print("✅ No known security vulnerabilities found")
                return True
        else:
            print("⚠️  Safety check failed (may not be installed)")
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️  Security check timed out")
        return True
    except Exception as e:
        print(f"⚠️  Security check error: {e}")
        return True

def generate_report():
    """Generate a comprehensive dependency report."""
    print("\n📋 Generating dependency report...")
    
    try:
        # Get list of installed packages
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            
            report = {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'total_packages': len(packages),
                'core_dependencies_status': {},
                'all_packages': {pkg['name']: pkg['version'] for pkg in packages}
            }
            
            # Check status of core dependencies
            for dep, expected_version in CORE_DEPENDENCIES.items():
                installed_version = report['all_packages'].get(dep.replace('-', '_'))
                if not installed_version:
                    installed_version = report['all_packages'].get(dep)
                
                report['core_dependencies_status'][dep] = {
                    'expected': expected_version,
                    'installed': installed_version,
                    'status': 'match' if installed_version == expected_version else 'mismatch' if installed_version else 'missing'
                }
            
            # Save report
            report_file = Path('dependency-report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Dependency report saved to {report_file}")
            return True
            
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 Python Dependencies Validation")
    print("=" * 40)
    
    # Track overall success
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    print("\n📦 Checking core dependencies...")
    # Check core dependencies
    for package, expected_version in CORE_DEPENDENCIES.items():
        success &= check_dependency(package, expected_version)
    
    # Run functionality tests
    success &= test_audio_processing()
    success &= test_tensorflow()
    
    # Security check
    check_security_vulnerabilities()
    
    # Generate report
    generate_report()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All core dependencies validated successfully!")
        print("🚀 System ready for audio processing")
    else:
        print("❌ Some dependencies have issues")
        print("🔧 Please check the output above and fix any problems")
        sys.exit(1)

if __name__ == "__main__":
    main()
