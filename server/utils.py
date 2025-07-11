"""
utils.py

This script analyzes an audio file and extracts useful metadata, including:
- Audio format
- Duration in seconds
- Bitrate
- Estimated tempo (BPM)
- Detected musical key

It uses librosa and pydub for audio analysis and handles errors gracefully with a fallback mechanism.
Run the script from the command line with a file path, and it outputs JSON-formatted metadata.
"""

import sys
import json
import logging
import re
from pathlib import Path

import librosa
import numpy as np
from os import path
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_audio_format(file_path):
    return path.splitext(file_path)[1][1:].lower()


def get_audio_data(file_path):
    return librosa.load(file_path, sr=None)


def get_audio_duration(audio_array, sample_rate):
    return librosa.get_duration(y=audio_array, sr=sample_rate)


def get_audio_bitrate(audio):
    return audio.frame_rate * audio.sample_width * audio.channels * 8


def get_audio_tempo(audio_array, sample_rate):
    onset_env = librosa.onset.onset_strength(y=audio_array, sr=sample_rate)
    return librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0]


def detect_key(audio_array, sample_rate):
    chroma = librosa.feature.chroma_cqt(y=audio_array, sr=sample_rate)
    chroma_sum = np.sum(chroma, axis=1)
    key_idx = np.argmax(chroma_sum)

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = key_names[key_idx]

    minor_chroma = librosa.feature.chroma_cqt(
        y=audio_array, sr=sample_rate, bins_per_octave=36)
    minor_sum = np.sum(minor_chroma[9:], axis=1) / np.sum(minor_chroma, axis=1)

    key_type = "minor" if np.mean(minor_sum) > 0.2 else "major"
    return f"{key} {key_type}"


def analyze_audio_file(file_path):
    """Analyze audio file and extract metadata.
    
    Args:
        file_path: Path to audio file (should be pre-validated)
        
    Returns:
        dict: Audio metadata or fallback data if analysis fails
    """
    safe_log_path = sanitize_file_path_for_logging(file_path)
    logger.info("Analyzing audio file: %s", safe_log_path)

    try:
        logger.info("Starting audio analysis...")
        format_type = get_audio_format(file_path)

        audio_array, sample_rate = get_audio_data(file_path)
        duration = int(get_audio_duration(audio_array, sample_rate))

        audio = AudioSegment.from_file(file_path)
        bitrate = int(get_audio_bitrate(audio))

        tempo = int(round(get_audio_tempo(audio_array, sample_rate)))
        key = detect_key(audio_array, sample_rate)

        info = {
            "format": format_type,
            "duration": duration,
            "bpm": tempo,
            "key": key,
            "bitrate": bitrate
        }

        logger.info("Successfully analyzed audio file: %s", safe_log_path)
        return info

    except Exception as error:
        logger.error("Error analyzing audio file %s: %s", safe_log_path, str(error))
        return fallback_audio_analysis(file_path)


def fallback_audio_analysis(file_path):
    """Fallback analysis when primary analysis fails.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        dict: Basic audio metadata with safe defaults
    """
    safe_log_path = sanitize_file_path_for_logging(file_path)
    logger.info("Using fallback analysis for: %s", safe_log_path)
    
    try:
        audio = AudioSegment.from_file(file_path)
        format_type = get_audio_format(file_path)

        return {
            "format": format_type,
            "duration": int(len(audio) / 1000),
            "bpm": 120,  # Safe default
            "key": "Unknown",
            "bitrate": int(get_audio_bitrate(audio))
        }

    except Exception as error:
        logger.error("Fallback analysis failed for %s: %s", safe_log_path, str(error))

        return {
            "format": "unknown",
            "duration": 0,
            "bpm": 0,
            "key": "Unknown",
            "bitrate": 0
        }


def is_valid_filepath(file_path):
    return path.exists(file_path)


def validate_audio_file_path(file_path):
    """Validate and sanitize audio file path.
    
    Args:
        file_path: Path to validate
        
    Returns:
        str: Sanitized path if valid, None if invalid
    """
    if not file_path or not isinstance(file_path, str):
        logger.error("Invalid file path type or empty path")
        return None
    
    try:
        # Convert to Path object for safe handling
        path_obj = Path(file_path).resolve()
        
        # Check if path exists and is a file
        if not path_obj.exists():
            logger.error("File does not exist: %s", file_path)
            return None
            
        if not path_obj.is_file():
            logger.error("Path is not a file: %s", file_path)
            return None
            
        # Check file extension is audio format
        allowed_extensions = {'.mp3', '.wav', '.flac', '.aiff', '.m4a', '.ogg'}
        if path_obj.suffix.lower() not in allowed_extensions:
            logger.error("Invalid audio file extension: %s", path_obj.suffix)
            return None
            
        # Check file size is reasonable (max 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if path_obj.stat().st_size > max_size:
            logger.error("File too large: %s bytes", path_obj.stat().st_size)
            return None
            
        return str(path_obj)
    except (OSError, ValueError) as e:
        logger.error("Path validation error: %s", str(e))
        return None


def sanitize_file_path_for_logging(file_path):
    """Sanitize file path for safe logging.
    
    Args:
        file_path: File path to sanitize
        
    Returns:
        str: Sanitized path safe for logging
    """
    if not file_path:
        return "unknown_file"
    
    try:
        path_obj = Path(file_path)
        # Only log the filename, not the full path
        return f"file_{hash(str(path_obj)) % 10000}_{path_obj.suffix}"
    except Exception:
        return "unknown_file"


def main():
    """Main function to handle command line execution."""
    try:
        if len(sys.argv) < 2:
            # Generic error message for client
            print(json.dumps({"error": "Audio analysis failed - insufficient parameters"}))
            sys.exit(1)

        # Validate and sanitize the file path
        file_path = validate_audio_file_path(sys.argv[1])
        if not file_path:
            # Generic error message for client (detailed error already logged)
            print(json.dumps({"error": "Audio analysis failed - file not accessible"}))
            sys.exit(1)

        # Analyze the validated file
        info = analyze_audio_file(file_path)
        print(json.dumps(info))
        sys.exit(0)
        
    except Exception as e:
        # Log detailed error server-side only
        logger.error("Unexpected error in main audio analysis: %s", str(e))
        # Generic error message for client
        print(json.dumps({"error": "Audio analysis failed - please check your file"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
