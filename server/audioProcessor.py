"""
audioProcessor.py

This script processes a music track to generate a remixed version with an optional shuffled intro and outro.

Features:
- Loads the input audio file ('audio.mp3').
- Uses Librosa to detect tempo (BPM) and beat positions.
- Optionally uses Madmom for more accurate tempo and beat tracking.
- Converts beat frames to actual time values.
- Separates audio into stems (vocals, drums, bass, other) using Spleeter (4 stems model).
- Identifies and selects the loudest segments from instrumental stems (bass, drums, other).
- Creates an intro by shuffling and stitching together the loudest instrumental segments.
- Optionally appends an outro using a similar shuffle method.
- Combines intro, original track, and optional outro into a single final remix.
- Saves the final output as 'output.mp3'.
- Also saves metadata (e.g., the shuffle order) in 'shuffle_info.json'.
"""

import sys
import os
import librosa
from pydub import AudioSegment
import json
import tempfile
import subprocess
import logging
import random
import stat
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_secure_temp_directory(prefix="airemixer_"):
    """
    Create a secure temporary directory with restricted permissions.
    
    Args:
        prefix: Prefix for the temporary directory name
    
    Returns:
        str: Path to the secure temporary directory
    
    Raises:
        OSError: If directory creation fails
    """
    try:
        # Create temporary directory with secure permissions
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        
        # Set restrictive permissions (owner read/write/execute only)
        # This is equivalent to chmod 700
        os.chmod(temp_dir, stat.S_IRWXU)
        
        logger.info("Created secure temporary directory: %s", temp_dir.split(os.sep)[-1])
        return temp_dir
        
    except OSError as e:
        logger.error("Failed to create secure temporary directory: %s", str(e))
        raise


def secure_cleanup_temp_directory(temp_dir):
    """
    Securely clean up temporary directory and all its contents.
    
    Args:
        temp_dir: Path to temporary directory to clean up
    
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    try:
        if not temp_dir or not os.path.exists(temp_dir):
            return True
            
        # Verify the directory is actually a temporary directory
        temp_prefix = os.path.basename(temp_dir).startswith("airemixer_")
        temp_parent = os.path.dirname(temp_dir) == tempfile.gettempdir()
        
        if not (temp_prefix and temp_parent):
            logger.warning("Refusing to delete non-temporary directory: %s", temp_dir)
            return False
        
        # Recursively remove all files and subdirectories
        shutil.rmtree(temp_dir, ignore_errors=False)
        logger.info("Successfully cleaned up temporary directory")
        return True
        
    except Exception as e:
        logger.error("Failed to clean up temporary directory: %s", str(e))
        return False


def validate_temp_file_path(file_path, allowed_temp_dir):
    """
    Validate that a file path is within the allowed temporary directory.
    
    Args:
        file_path: Path to validate
        allowed_temp_dir: The allowed temporary directory
    
    Returns:
        bool: True if path is safe, False otherwise
    """
    try:
        # Resolve both paths to absolute paths
        resolved_file = Path(file_path).resolve()
        resolved_temp = Path(allowed_temp_dir).resolve()
        
        # Check if the file is within the allowed temp directory
        try:
            resolved_file.relative_to(resolved_temp)
            return True
        except ValueError:
            logger.warning("File path outside allowed temporary directory: %s", file_path)
            return False
            
    except Exception as e:
        logger.error("Error validating temp file path: %s", str(e))
        return False


def check_dependencies():
    """Check if required dependencies are installed without auto-installing them."""
    missing_deps = []
    
    # Check for madmom
    try:
        import madmom
        logger.info("madmom dependency found")
    except ImportError:
        missing_deps.append("madmom")
    
    # Check for spleeter
    try:
        from spleeter.separator import Separator
        logger.info("spleeter dependency found")
    except ImportError:
        missing_deps.append("spleeter")
    
    # Check for other required packages
    required_packages = ["librosa", "numpy", "pydub"]
    for package in required_packages:
        try:
            __import__(package)
            logger.info("%s dependency found", package)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}. Please install them manually using: pip install {' '.join(missing_deps)}"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    logger.info("All required dependencies are available")


# Check dependencies at startup
check_dependencies()

# Now safely import the dependencies we've verified
try:
    import madmom
    from spleeter.separator import Separator
    logger.info("Successfully imported all audio processing dependencies")
except ImportError as e:
    logger.error("Failed to import dependencies after verification: %s", str(e))
    raise


def detect_tempo_and_beats(audio_path, method="auto"):
    logger.info("Detecting tempo and beats using %s method", method)

    if method in ("librosa", "auto"):
        try:
            y, sr = librosa.load(audio_path, sr=None)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)

            if len(beats) > 0 and tempo > 0:
                logger.info(
                    "Librosa detected tempo: %s BPM with %s beats", tempo, len(beats))
                return tempo, beat_times
            elif method == "librosa":
                logger.warning(
                    "Librosa beat detection failed, but was explicitly requested")
                return None, None
        except Exception as e:
            logger.error("Error in librosa beat detection: %s", str(e))
            if method == "librosa":
                return None, None

    if method in ("madmom", "auto"):
        try:
            from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
            from madmom.features.tempo import TempoEstimationProcessor

            proc = RNNBeatProcessor()(audio_path)
            beats = BeatTrackingProcessor(fps=100)(proc)
            tempo_proc = TempoEstimationProcessor(fps=100)(proc)
            tempo = tempo_proc[0][0]

            logger.info(
                "Madmom detected tempo: %s BPM with %s beats", tempo, len(beats))
            return tempo, beats
        except Exception as e:
            logger.error("Error in madmom beat detection: %s", str(e))

    logger.warning("Beat detection failed with all methods")
    return None, None


def separate_audio_components(audio_path, output_dir):
    logger.info("Starting audio separation with Spleeter")

    # Validate that output_dir is within allowed temporary directory
    if not validate_temp_file_path(output_dir, tempfile.gettempdir()):
        logger.error("Output directory not in allowed temporary location")
        return None

    try:
        separator = Separator('spleeter:4stems')
        main_song = AudioSegment.from_file(audio_path)
        separator.separate_to_file(audio_path, output_dir)

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        component_dir = os.path.join(output_dir, base_name)

        components = {
            'vocals': os.path.join(component_dir, 'vocals.wav'),
            'drums': os.path.join(component_dir, 'drums.wav'),
            'bass': os.path.join(component_dir, 'bass.wav'),
            'other': os.path.join(component_dir, 'other.wav')
        }

        # Validate all component file paths are within the secure temp directory
        for component, path in components.items():
            if not validate_temp_file_path(path, output_dir):
                logger.error("Component file path validation failed for %s: %s", component, path)
                return None
                
            if not os.path.exists(path):
                logger.warning("Component %s file not found at %s", component, path)

        logger.info("Audio separation completed successfully")
        return [components, main_song]

    except Exception as e:
        logger.error("Error during audio separation: %s", str(e))
        return None


def pick_loudest_bars(stem, beats_ms, bars=4, beats_per_bar=4):
    total_beats = len(beats_ms)
    window = beats_per_bar * bars
    max_rms = -1
    pick_start = 0
    if total_beats < window + 1:
        return stem
    for i in range(total_beats - window):
        start_ms = int(beats_ms[i])
        end_ms = int(beats_ms[i + window])
        segment = stem[start_ms:end_ms]
        rms = segment.rms
        if rms > max_rms:
            max_rms = rms
            pick_start = i
    start_ms = int(beats_ms[pick_start])
    end_ms = int(beats_ms[pick_start + window])
    return stem[start_ms:end_ms]


def create_extended_mix(components, output_path, intro_bars, outro_bars, _preserve_vocals, _tempo, beat_times, main_song):
    logger.info(
        "Creating extended mix with %s bars intro and %s bars outro", intro_bars, outro_bars)

    try:
        # Validate all component file paths for security
        for component_name, component_path in components.items():
            if not os.path.exists(component_path):
                logger.error("Component file missing: %s at %s", component_name, component_path)
                return False
            
            # Ensure component files are readable
            if not os.access(component_path, os.R_OK):
                logger.error("Component file not readable: %s", component_name)
                return False

        beats_per_bar = 4
        intro_beats = intro_bars * beats_per_bar
        outro_beats = outro_bars * beats_per_bar

        if len(beat_times) < (intro_beats + outro_beats + 8):
            logger.warning(
                "Not enough beats detected (%s) for requested extension", len(beat_times))
            return False

        version = 1
        if "_v" in output_path:
            try:
                version = int(output_path.split("_v")[-1].split(".")[0])
            except (ValueError, IndexError):
                pass

        # Securely load audio components with error handling
        try:
            drums = AudioSegment.from_file(components['drums'])
            bass = AudioSegment.from_file(components['bass']) + 12
            other = AudioSegment.from_file(components['other']).apply_gain(9)
            vocals = AudioSegment.from_file(components['vocals'])
        except Exception as e:
            logger.error("Failed to load audio components: %s", str(e))
            return False

        beat_times_ms = [t * 1000 for t in beat_times]

        full_intro_drums = pick_loudest_bars(
            drums, beat_times_ms, bars=intro_bars)
        _unused_full_intro_bass = pick_loudest_bars(
            bass, beat_times_ms, bars=intro_bars)
        full_intro_other = pick_loudest_bars(
            other, beat_times_ms, bars=intro_bars)
        intro_vocals = pick_loudest_bars(
            vocals, beat_times_ms, bars=intro_bars)

        # Use deterministic seeding for reproducible results
        random.seed(version * 42)

        intro_labels = ['drums', 'other', 'drums', 'vocals']
        intro_segments = [full_intro_drums,
                          full_intro_other, full_intro_drums, intro_vocals]
        intro_zipped = list(zip(intro_labels, intro_segments))
        random.shuffle(intro_zipped)
        intro_components = [seg for (_, seg) in intro_zipped]
        shuffled_intro_order = [label for (label, _) in intro_zipped]
        
        # Reset random seed for security
        random.seed()

        full_intro = sum(intro_components).fade_in(2000)
        extended_mix = full_intro.append(main_song, crossfade=500)

        # Validate output path before writing
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logger.error("Output directory does not exist: %s", output_dir)
            return False
            
        if not os.access(output_dir, os.W_OK):
            logger.error("No write permission for output directory: %s", output_dir)
            return False

        # Securely export the final mix
        try:
            file_format = os.path.splitext(output_path)[1][1:].lower()
            if file_format not in ['mp3', 'wav', 'flac', 'aiff']:
                logger.error("Unsupported output format: %s", file_format)
                return False
                
            extended_mix.export(output_path, format=file_format)
            logger.info("Extended mix created successfully and saved to %s", output_path)
            
            # Verify the output file was created successfully
            if not os.path.exists(output_path):
                logger.error("Output file was not created successfully")
                return False
                
        except Exception as e:
            logger.error("Failed to export extended mix: %s", str(e))
            return False

        return True

    except Exception as e:
        logger.error("Error in create_extended_mix: %s", str(e))
        return False


def process_audio(input_path, output_path, intro_bars=16, outro_bars=16, preserve_vocals=True, beat_detection="auto"):
    """
    Process audio with enhanced security validation and secure temporary directories.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        intro_bars: Number of intro bars (1-64)
        outro_bars: Number of outro bars (1-64)
        preserve_vocals: Whether to preserve vocals
        beat_detection: Beat detection method ('auto', 'librosa', 'madmom')
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Sanitize input_path for logging (remove sensitive information)
    safe_input_log = f"file_{hash(input_path) % 10000}"
    logger.info("Starting audio processing: %s", safe_input_log)
    logger.info(
        "Parameters: intro_bars=%s, outro_bars=%s, preserve_vocals=%s, beat_detection=%s", 
        intro_bars, outro_bars, preserve_vocals, beat_detection)

    temp_dir = None
    try:
        # Additional runtime validation for critical parameters
        if not isinstance(intro_bars, int) or intro_bars < 1 or intro_bars > 64:
            logger.error("Invalid intro_bars parameter: %s", intro_bars)
            return False
            
        if not isinstance(outro_bars, int) or outro_bars < 1 or outro_bars > 64:
            logger.error("Invalid outro_bars parameter: %s", outro_bars)
            return False
            
        if beat_detection not in ['auto', 'librosa', 'madmom']:
            logger.error("Invalid beat_detection parameter: %s", beat_detection)
            return False

        # Validate file paths exist and are accessible
        if not os.path.exists(input_path):
            logger.error("Input file does not exist")
            return False
            
        if not os.access(input_path, os.R_OK):
            logger.error("Input file is not readable")
            return False

        # Ensure preserve_vocals is boolean
        preserve_vocals = bool(preserve_vocals)

        # Create secure temporary directory with proper permissions
        temp_dir = create_secure_temp_directory("airemixer_audio_")
        logger.info("Using secure temporary directory for processing")

        tempo, beat_times = detect_tempo_and_beats(
            input_path, method=beat_detection)
        if tempo is None or beat_times is None or len(beat_times) == 0:
            logger.error("Beat detection failed, cannot proceed")
            return False

        components, main_song = separate_audio_components(
            input_path, temp_dir)
        if components is None:
            logger.error("Audio separation failed, cannot proceed")
            return False

        success = create_extended_mix(
            components,
            output_path,
            intro_bars,
            outro_bars,
            preserve_vocals,
            tempo,
            beat_times,
            main_song
        )

        return success

    except Exception as e:
        logger.error("Error in audio processing: %s", str(e))
        return False
    
    finally:
        # Always clean up temporary directory
        if temp_dir:
            cleanup_success = secure_cleanup_temp_directory(temp_dir)
            if not cleanup_success:
                logger.warning("Failed to clean up temporary directory: %s", temp_dir)


def validate_input_parameters(input_path, output_path, intro_bars, outro_bars, preserve_vocals, beat_detection):
    """
    Validate and sanitize all input parameters for security.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        intro_bars: Number of intro bars
        outro_bars: Number of outro bars
        preserve_vocals: Boolean for vocal preservation
        beat_detection: Beat detection method
    
    Returns:
        dict: Validated parameters or None if validation fails
    """
    import os
    from pathlib import Path
    
    # Validate input file path
    if not input_path or not isinstance(input_path, str):
        logger.error("Invalid input path: must be a non-empty string")
        return None
    
    try:
        input_path_obj = Path(input_path).resolve()
        if not input_path_obj.exists():
            logger.error("Input file does not exist: %s", input_path)
            return None
        
        if not input_path_obj.is_file():
            logger.error("Input path is not a file: %s", input_path)
            return None
        
        # Check file extension
        allowed_extensions = {'.mp3', '.wav', '.flac', '.aiff', '.m4a'}
        if input_path_obj.suffix.lower() not in allowed_extensions:
            logger.error("Invalid input file extension: %s", input_path_obj.suffix)
            return None
        
        # Check file size (max 500MB for security)
        max_size = int(os.environ.get('MAX_AUDIO_FILE_SIZE', 500 * 1024 * 1024))
        if input_path_obj.stat().st_size > max_size:
            logger.error("Input file too large: %s bytes (max: %s)", 
                        input_path_obj.stat().st_size, max_size)
            return None
        
        # Additional security: Check for symbolic links
        if input_path_obj.is_symlink():
            logger.error("Symbolic links not allowed for input files")
            return None
            
    except (OSError, ValueError) as e:
        logger.error("Input path validation error: %s", str(e))
        return None
    
    # Validate output file path
    if not output_path or not isinstance(output_path, str):
        logger.error("Invalid output path: must be a non-empty string")
        return None
    
    try:
        output_path_obj = Path(output_path).resolve()
        output_dir = output_path_obj.parent
        
        # Check if output directory exists
        if not output_dir.exists():
            logger.error("Output directory does not exist: %s", output_dir)
            return None
        
        # Check write permissions
        if not os.access(output_dir, os.W_OK):
            logger.error("No write permission for output directory: %s", output_dir)
            return None
        
        # Check output file extension
        if output_path_obj.suffix.lower() not in allowed_extensions:
            logger.error("Invalid output file extension: %s", output_path_obj.suffix)
            return None
        
        # Additional security: Prevent overwriting critical system files
        if str(output_path_obj).startswith(('/bin', '/sbin', '/usr/bin', '/usr/sbin', '/etc')):
            logger.error("Output path in restricted system directory")
            return None
            
    except (OSError, ValueError) as e:
        logger.error("Output path validation error: %s", str(e))
        return None
    
    # Validate intro_bars
    try:
        intro_bars = int(intro_bars)
        if intro_bars < 1 or intro_bars > 64:
            logger.error("Invalid intro_bars: must be between 1 and 64, got %s", intro_bars)
            return None
    except (ValueError, TypeError):
        logger.error("Invalid intro_bars: must be an integer, got %s", intro_bars)
        return None
    
    # Validate outro_bars
    try:
        outro_bars = int(outro_bars)
        if outro_bars < 1 or outro_bars > 64:
            logger.error("Invalid outro_bars: must be between 1 and 64, got %s", outro_bars)
            return None
    except (ValueError, TypeError):
        logger.error("Invalid outro_bars: must be an integer, got %s", outro_bars)
        return None
    
    # Validate preserve_vocals
    if isinstance(preserve_vocals, str):
        preserve_vocals_str = preserve_vocals.lower().strip()
        if preserve_vocals_str in ('true', '1', 'yes', 'on'):
            preserve_vocals = True
        elif preserve_vocals_str in ('false', '0', 'no', 'off'):
            preserve_vocals = False
        else:
            logger.error("Invalid preserve_vocals: must be true/false, got %s", preserve_vocals)
            return None
    else:
        preserve_vocals = bool(preserve_vocals)
    
    # Validate beat_detection method
    if not isinstance(beat_detection, str):
        logger.error("Invalid beat_detection: must be a string, got %s", type(beat_detection))
        return None
    
    beat_detection = beat_detection.lower().strip()
    allowed_methods = ['auto', 'librosa', 'madmom']
    if beat_detection not in allowed_methods:
        logger.error("Invalid beat_detection: must be one of %s, got %s", allowed_methods, beat_detection)
        return None
    
    return {
        'input_path': str(input_path_obj),
        'output_path': str(output_path_obj),
        'intro_bars': intro_bars,
        'outro_bars': outro_bars,
        'preserve_vocals': preserve_vocals,
        'beat_detection': beat_detection
    }


def main():
    """Main function to handle command line execution with comprehensive input validation."""
    if len(sys.argv) < 3:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: python audioProcessor.py <input_path> <output_path> [intro_bars] [outro_bars] [preserve_vocals] [beat_detection]"
        }))
        sys.exit(1)

    try:
        # Extract command line arguments
        raw_input_path = sys.argv[1]
        raw_output_path = sys.argv[2]
        raw_intro_bars = sys.argv[3] if len(sys.argv) > 3 else 16
        raw_outro_bars = sys.argv[4] if len(sys.argv) > 4 else 16
        raw_preserve_vocals = sys.argv[5] if len(sys.argv) > 5 else True
        raw_beat_detection = sys.argv[6] if len(sys.argv) > 6 else "auto"

        # Validate and sanitize all input parameters
        validated_params = validate_input_parameters(
            raw_input_path,
            raw_output_path,
            raw_intro_bars,
            raw_outro_bars,
            raw_preserve_vocals,
            raw_beat_detection
        )

        if not validated_params:
            print(json.dumps({
                "status": "error",
                "message": "Invalid input parameters provided"
            }))
            sys.exit(1)

        # Log the validated parameters (without sensitive path information)
        logger.info("Processing audio with validated parameters:")
        logger.info("- Intro bars: %s", validated_params['intro_bars'])
        logger.info("- Outro bars: %s", validated_params['outro_bars'])
        logger.info("- Preserve vocals: %s", validated_params['preserve_vocals'])
        logger.info("- Beat detection: %s", validated_params['beat_detection'])

        # Process audio with validated parameters
        processing_success = process_audio(
            validated_params['input_path'],
            validated_params['output_path'],
            validated_params['intro_bars'],
            validated_params['outro_bars'],
            validated_params['preserve_vocals'],
            validated_params['beat_detection']
        )

        if processing_success:
            print(json.dumps({
                "status": "success",
                "output_path": validated_params['output_path']
            }))
            sys.exit(0)
        else:
            print(json.dumps({
                "status": "error",
                "message": "Audio processing failed"
            }))
            sys.exit(1)

    except Exception as e:
        logger.error("Unexpected error in main: %s", str(e))
        print(json.dumps({
            "status": "error",
            "message": "Unexpected error occurred during processing"
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
