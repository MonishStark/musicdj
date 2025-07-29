"""
audio_optimization_integration.py

Integration script to enable memory-optimized audio processing in the Music DJ Feature application.
This script provides backward-compatible wrappers and environment detection.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Union

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

logger = logging.getLogger(__name__)

class AudioProcessingOptimizer:
    """
    Main integration class for memory-optimized audio processing.
    Provides backward compatibility while enabling optimization features.
    """
    
    def __init__(self, enable_optimization: bool = True, environment: Optional[str] = None):
        """
        Initialize the audio processing optimizer.
        
        Args:
            enable_optimization: Whether to use optimized processing
            environment: Target environment (auto-detected if None)
        """
        self.enable_optimization = enable_optimization
        self.environment = environment or self._detect_environment()
        
        # Try to import optimized modules
        self.optimized_available = self._check_optimized_availability()
        
        if self.optimized_available and enable_optimization:
            self._setup_optimized_processing()
        else:
            self._setup_fallback_processing()
    
    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        if os.environ.get('AZURE_FUNCTIONS_ENVIRONMENT'):
            return "azure_functions"
        elif os.environ.get('CONTAINER_APP_NAME'):
            return "azure_container_apps"
        elif os.environ.get('NODE_ENV') == 'development':
            return "development"
        elif os.environ.get('NODE_ENV') == 'test':
            return "testing"
        else:
            return "production"
    
    def _check_optimized_availability(self) -> bool:
        """Check if optimized modules are available."""
        try:
            # Check for required dependencies
            import gc
            
            # Try to import optimized modules
            if os.path.exists(os.path.join(current_dir, 'utils_optimized.py')):
                return True
            
            return False
        except ImportError as e:
            logger.warning(f"Optimized modules not available: {e}")
            return False
    
    def _setup_optimized_processing(self):
        """Setup optimized processing modules."""
        try:
            from utils_optimized import analyze_audio_file_optimized
            from audioProcessor_optimized import process_audio_memory_optimized
            from audio_processing_config import get_processing_config
            
            self.analyze_audio = analyze_audio_file_optimized
            self.process_audio = process_audio_memory_optimized
            self.get_config = get_processing_config
            
            logger.info(f"Optimized audio processing enabled for {self.environment}")
            
        except ImportError as e:
            logger.warning(f"Failed to setup optimized processing: {e}")
            self._setup_fallback_processing()
    
    def _setup_fallback_processing(self):
        """Setup fallback to original processing."""
        try:
            from utils import analyze_audio_file
            from audioProcessor import process_audio
            
            self.analyze_audio = self._wrap_original_analyze
            self.process_audio = self._wrap_original_process
            self.get_config = lambda env=None: {"processing_quality": "standard"}
            
            logger.info(f"Using original audio processing for {self.environment}")
            
        except ImportError as e:
            logger.error(f"Failed to setup fallback processing: {e}")
            raise
    
    def _wrap_original_analyze(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Wrapper for original analyze_audio_file function."""
        from utils import analyze_audio_file
        
        try:
            result = analyze_audio_file(file_path)
            
            # Ensure consistent return format
            if isinstance(result, dict):
                return result
            else:
                return {"error": "Invalid analysis result"}
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
    
    def _wrap_original_process(self, input_path: str, output_path: str, 
                             intro_bars: int = 16, outro_bars: int = 16, 
                             preserve_vocals: bool = True, beat_detection: str = "auto",
                             **kwargs) -> bool:
        """Wrapper for original process_audio function."""
        from audioProcessor import process_audio
        
        try:
            return process_audio(
                input_path, output_path, intro_bars, outro_bars,
                preserve_vocals, beat_detection
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
    
    def analyze_audio_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze audio file using optimal method.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional arguments (quality, chunk_duration, etc.)
        
        Returns:
            Dictionary containing analysis results
        """
        if self.optimized_available and self.enable_optimization:
            # Use optimized analysis with environment-specific settings
            quality = kwargs.get('quality', 'balanced')
            chunk_duration = kwargs.get('chunk_duration', 30.0)
            
            # Adjust settings based on environment
            if self.environment in ['azure_functions', 'testing']:
                quality = 'fast'
                chunk_duration = 60.0
            
            return self.analyze_audio(file_path, quality=quality, chunk_duration=chunk_duration)
        else:
            # Use original analysis
            return self.analyze_audio(file_path)
    
    def process_audio_file(self, input_path: str, output_path: str, **kwargs) -> bool:
        """
        Process audio file using optimal method.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            **kwargs: Processing parameters
        
        Returns:
            True if successful, False otherwise
        """
        # Extract parameters with defaults
        intro_bars = kwargs.get('intro_bars', 16)
        outro_bars = kwargs.get('outro_bars', 16)
        preserve_vocals = kwargs.get('preserve_vocals', True)
        beat_detection = kwargs.get('beat_detection', 'auto')
        
        if self.optimized_available and self.enable_optimization:
            # Use optimized processing with environment-specific settings
            processing_quality = kwargs.get('processing_quality', 'balanced')
            max_memory_gb = kwargs.get('max_memory_gb', 4.0)
            
            # Adjust settings based on environment
            if self.environment == 'azure_functions':
                processing_quality = 'fast'
                max_memory_gb = 1.5
            elif self.environment == 'testing':
                processing_quality = 'fast'
                max_memory_gb = 1.0
            
            return self.process_audio(
                input_path, output_path, intro_bars, outro_bars,
                preserve_vocals, beat_detection, processing_quality, max_memory_gb
            )
        else:
            # Use original processing
            return self.process_audio(
                input_path, output_path, intro_bars, outro_bars,
                preserve_vocals, beat_detection
            )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and configuration."""
        status = {
            "optimization_enabled": self.enable_optimization,
            "optimized_available": self.optimized_available,
            "environment": self.environment,
            "memory_usage": self.get_memory_usage()
        }
        
        if self.optimized_available and self.enable_optimization:
            try:
                config = self.get_config(self.environment)
                status["configuration"] = config.to_dict() if hasattr(config, 'to_dict') else config
            except Exception as e:
                status["configuration_error"] = str(e)
        
        return status

# Global optimizer instance
_optimizer = None

def get_audio_optimizer(enable_optimization: bool = True, environment: Optional[str] = None) -> AudioProcessingOptimizer:
    """Get or create global audio processing optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = AudioProcessingOptimizer(enable_optimization, environment)
    return _optimizer

def analyze_audio_file_integrated(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Integrated audio analysis function that automatically uses optimization when available.
    
    This function provides a drop-in replacement for the original analyze_audio_file function
    with automatic optimization and environment detection.
    """
    optimizer = get_audio_optimizer()
    return optimizer.analyze_audio_file(file_path, **kwargs)

def process_audio_integrated(input_path: str, output_path: str, **kwargs) -> bool:
    """
    Integrated audio processing function that automatically uses optimization when available.
    
    This function provides a drop-in replacement for the original process_audio function
    with automatic optimization and environment detection.
    """
    optimizer = get_audio_optimizer()
    return optimizer.process_audio_file(input_path, output_path, **kwargs)

def get_optimization_info() -> Dict[str, Any]:
    """Get information about current optimization status."""
    try:
        optimizer = get_audio_optimizer()
        return optimizer.get_optimization_status()
    except Exception as e:
        return {"error": str(e), "optimization_available": False}

# Backward compatibility aliases
analyze_audio_file = analyze_audio_file_integrated
process_audio = process_audio_integrated

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            info = get_optimization_info()
            print("Audio Processing Optimization Status:")
            print(json.dumps(info, indent=2))
        
        elif command == "test_analyze" and len(sys.argv) > 2:
            file_path = sys.argv[2]
            if os.path.exists(file_path):
                result = analyze_audio_file_integrated(file_path)
                print("Analysis Result:")
                print(json.dumps(result, indent=2))
            else:
                print(f"File not found: {file_path}")
        
        elif command == "test_process" and len(sys.argv) > 3:
            input_path = sys.argv[2]
            output_path = sys.argv[3]
            if os.path.exists(input_path):
                success = process_audio_integrated(input_path, output_path)
                print(f"Processing {'succeeded' if success else 'failed'}")
            else:
                print(f"Input file not found: {input_path}")
        
        else:
            print("Usage: python audio_optimization_integration.py [status|test_analyze|test_process] [file_path] [output_path]")
    else:
        print("Audio Processing Optimization Integration")
        print("This module provides optimized audio processing with automatic fallback.")
        print("\nCommands:")
        print("  status                              - Show optimization status")
        print("  test_analyze <file_path>           - Test audio analysis")
        print("  test_process <input> <output>      - Test audio processing")
        
        # Show current status
        info = get_optimization_info()
        print(f"\nCurrent Status: {'Optimized' if info.get('optimization_enabled') else 'Standard'}")
        print(f"Environment: {info.get('environment', 'unknown')}")
