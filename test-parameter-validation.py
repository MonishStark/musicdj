#!/usr/bin/env python3
"""
Test script to verify parameter validation and sanitization
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

# Import validation functions
try:
    from audioProcessor import (
        validate_file_path, 
        validate_integer_parameter, 
        validate_boolean_parameter,
        validate_beat_detection_method,
        validate_and_sanitize_parameters
    )
    from utils import validate_audio_file_path
    print("✓ Successfully imported validation functions")
except ImportError as e:
    print(f"✗ Failed to import validation functions: {e}")
    sys.exit(1)

def test_file_path_validation():
    """Test file path validation functions"""
    print("\n--- Testing File Path Validation ---")
    
    # Test non-existent file
    result = validate_file_path("/nonexistent/file.mp3")
    assert result is None, "Should reject non-existent file"
    print("✓ Rejects non-existent files")
    
    # Test invalid file type
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = tmp.name
    
    try:
        result = validate_file_path(tmp_path)
        assert result is None, "Should reject non-audio files"
        print("✓ Rejects non-audio file extensions")
    finally:
        os.unlink(tmp_path)
    
    # Test valid audio file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"fake mp3 content")
        tmp_path = tmp.name
    
    try:
        result = validate_file_path(tmp_path)
        assert result == str(Path(tmp_path).resolve()), "Should accept valid audio file"
        print("✓ Accepts valid audio files")
    finally:
        os.unlink(tmp_path)
    
    # Test None input
    result = validate_file_path(None)
    assert result is None, "Should reject None input"
    print("✓ Rejects None input")
    
    # Test empty string
    result = validate_file_path("")
    assert result is None, "Should reject empty string"
    print("✓ Rejects empty string")

def test_integer_parameter_validation():
    """Test integer parameter validation"""
    print("\n--- Testing Integer Parameter Validation ---")
    
    # Test valid integers
    assert validate_integer_parameter("16", "test") == 16
    assert validate_integer_parameter(32, "test") == 32
    assert validate_integer_parameter("8", "test", 1, 64) == 8
    print("✓ Accepts valid integers")
    
    # Test out of range
    assert validate_integer_parameter("100", "test", 1, 64) is None
    assert validate_integer_parameter("0", "test", 1, 64) is None
    print("✓ Rejects out-of-range values")
    
    # Test invalid input
    assert validate_integer_parameter("abc", "test") is None
    assert validate_integer_parameter(None, "test") is None
    assert validate_integer_parameter("", "test") is None
    print("✓ Rejects invalid input")
    
    # Test injection attempts
    assert validate_integer_parameter("16; rm -rf /", "test") is None
    assert validate_integer_parameter("16 && echo hacked", "test") is None
    print("✓ Rejects injection attempts")

def test_boolean_parameter_validation():
    """Test boolean parameter validation"""
    print("\n--- Testing Boolean Parameter Validation ---")
    
    # Test valid true values
    assert validate_boolean_parameter("true", "test") == True
    assert validate_boolean_parameter("True", "test") == True
    assert validate_boolean_parameter("1", "test") == True
    assert validate_boolean_parameter("yes", "test") == True
    assert validate_boolean_parameter(True, "test") == True
    print("✓ Accepts valid true values")
    
    # Test valid false values
    assert validate_boolean_parameter("false", "test") == False
    assert validate_boolean_parameter("False", "test") == False
    assert validate_boolean_parameter("0", "test") == False
    assert validate_boolean_parameter("no", "test") == False
    assert validate_boolean_parameter(False, "test") == False
    print("✓ Accepts valid false values")
    
    # Test invalid values
    assert validate_boolean_parameter("maybe", "test") is None
    assert validate_boolean_parameter("2", "test") is None
    assert validate_boolean_parameter(None, "test") is None
    print("✓ Rejects invalid values")

def test_beat_detection_validation():
    """Test beat detection method validation"""
    print("\n--- Testing Beat Detection Validation ---")
    
    # Test valid methods
    assert validate_beat_detection_method("auto") == "auto"
    assert validate_beat_detection_method("librosa") == "librosa"
    assert validate_beat_detection_method("madmom") == "madmom"
    assert validate_beat_detection_method("AUTO") == "auto"  # Case insensitive
    print("✓ Accepts valid methods")
    
    # Test invalid methods
    assert validate_beat_detection_method("invalid") is None
    assert validate_beat_detection_method("") is None
    assert validate_beat_detection_method(None) is None
    assert validate_beat_detection_method("auto; rm -rf /") is None
    print("✓ Rejects invalid methods")

def test_full_parameter_validation():
    """Test complete parameter validation"""
    print("\n--- Testing Full Parameter Validation ---")
    
    # Create a temporary audio file for testing
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"fake mp3 content")
        input_file = tmp.name
    
    output_dir = tempfile.mkdtemp()
    output_file = os.path.join(output_dir, "output.mp3")
    
    try:
        # Test valid parameters
        args = ['audioProcessor.py', input_file, output_file, '16', '16', 'true', 'auto']
        result = validate_and_sanitize_parameters(args)
        
        assert result is not None, "Should accept valid parameters"
        assert result['intro_bars'] == 16
        assert result['outro_bars'] == 16
        assert result['preserve_vocals'] == True
        assert result['beat_detection'] == 'auto'
        print("✓ Accepts complete valid parameter set")
        
        # Test minimal parameters (should use defaults)
        args = ['audioProcessor.py', input_file, output_file]
        result = validate_and_sanitize_parameters(args)
        
        assert result is not None, "Should accept minimal parameters"
        assert result['intro_bars'] == 16  # Default
        assert result['outro_bars'] == 16  # Default
        assert result['preserve_vocals'] == True  # Default
        assert result['beat_detection'] == 'auto'  # Default
        print("✓ Uses appropriate defaults")
        
        # Test insufficient parameters
        args = ['audioProcessor.py', input_file]
        result = validate_and_sanitize_parameters(args)
        assert result is None, "Should reject insufficient parameters"
        print("✓ Rejects insufficient parameters")
        
        # Test invalid file
        args = ['audioProcessor.py', '/nonexistent/file.mp3', output_file]
        result = validate_and_sanitize_parameters(args)
        assert result is None, "Should reject invalid input file"
        print("✓ Rejects invalid input file")
        
    finally:
        # Cleanup
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_dir):
            os.rmdir(output_dir)

def test_security_cases():
    """Test security-related edge cases"""
    print("\n--- Testing Security Cases ---")
    
    # Test path traversal attempts
    result = validate_file_path("../../../etc/passwd")
    assert result is None, "Should reject path traversal"
    print("✓ Rejects path traversal attempts")
    
    # Test command injection attempts in parameters
    result = validate_integer_parameter("16; rm -rf /", "test")
    assert result is None, "Should reject command injection"
    print("✓ Rejects command injection in integers")
    
    result = validate_beat_detection_method("auto && rm -rf /")
    assert result is None, "Should reject command injection"
    print("✓ Rejects command injection in method selection")
    
    # Test extremely large numbers
    result = validate_integer_parameter("999999999999", "test", 1, 64)
    assert result is None, "Should reject extremely large numbers"
    print("✓ Rejects extremely large numbers")

def main():
    """Run all tests"""
    print("🧪 Testing Parameter Validation and Sanitization")
    print("=" * 50)
    
    try:
        test_file_path_validation()
        test_integer_parameter_validation()
        test_boolean_parameter_validation()
        test_beat_detection_validation()
        test_full_parameter_validation()
        test_security_cases()
        
        print("\n" + "=" * 50)
        print("🎉 All parameter validation tests passed!")
        print("✓ File path validation working correctly")
        print("✓ Integer parameter validation working correctly")
        print("✓ Boolean parameter validation working correctly")
        print("✓ Beat detection method validation working correctly")
        print("✓ Full parameter validation working correctly")
        print("✓ Security measures working correctly")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
