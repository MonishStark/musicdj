"""
audio_processing_config.py

Configuration management for memory-optimized audio processing.
Provides centralized settings for different deployment environments and use cases.
"""

import os
import json
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration class for audio processing optimization."""
    
    # Memory management
    max_memory_gb: float = 4.0
    memory_safety_margin: float = 0.5  # GB to keep free
    enable_gc_optimization: bool = True
    
    # Quality settings
    processing_quality: str = "balanced"  # fast, balanced, high
    analysis_quality: str = "balanced"
    
    # Performance tuning
    chunk_duration: float = 30.0  # seconds
    enable_parallel_processing: bool = True
    max_concurrent_operations: int = 2
    
    # File handling
    cleanup_temp_files: bool = True
    preserve_intermediate_files: bool = False
    temp_dir_prefix: str = "music_dj_processing_"
    
    # Audio processing
    default_sample_rate: int = 44100
    separation_quality: int = 4  # Number of stems
    enable_beat_caching: bool = True
    
    # Streaming integration
    streaming_chunk_size: int = 1024 * 1024  # 1MB chunks
    enable_progress_callbacks: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_memory_gb': self.max_memory_gb,
            'memory_safety_margin': self.memory_safety_margin,
            'enable_gc_optimization': self.enable_gc_optimization,
            'processing_quality': self.processing_quality,
            'analysis_quality': self.analysis_quality,
            'chunk_duration': self.chunk_duration,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_concurrent_operations': self.max_concurrent_operations,
            'cleanup_temp_files': self.cleanup_temp_files,
            'preserve_intermediate_files': self.preserve_intermediate_files,
            'temp_dir_prefix': self.temp_dir_prefix,
            'default_sample_rate': self.default_sample_rate,
            'separation_quality': self.separation_quality,
            'enable_beat_caching': self.enable_beat_caching,
            'streaming_chunk_size': self.streaming_chunk_size,
            'enable_progress_callbacks': self.enable_progress_callbacks
        }

class ConfigManager:
    """Manages configuration for audio processing optimization."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), 'audio_processing_config.json'
        )
        self.config = self._load_or_create_config()
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities for automatic configuration."""
        try:
            # Get system memory
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # Detect if running in containerized environment
            is_containerized = os.path.exists('/.dockerenv') or os.environ.get('KUBERNETES_SERVICE_HOST')
            
            # Detect cloud environment
            cloud_provider = None
            if os.environ.get('AZURE_CLIENT_ID'):
                cloud_provider = 'azure'
            elif os.environ.get('AWS_EXECUTION_ENV'):
                cloud_provider = 'aws'
            elif os.environ.get('GOOGLE_CLOUD_PROJECT'):
                cloud_provider = 'gcp'
            
            return {
                'memory_gb': memory_gb,
                'cpu_count': cpu_count,
                'is_containerized': is_containerized,
                'cloud_provider': cloud_provider
            }
        except Exception:
            # Fallback to conservative defaults
            return {
                'memory_gb': 4.0,
                'cpu_count': 2,
                'is_containerized': False,
                'cloud_provider': None
            }
    
    def _get_optimal_config(self, system_info: Dict[str, Any]) -> ProcessingConfig:
        """Generate optimal configuration based on system capabilities."""
        memory_gb = system_info['memory_gb']
        cpu_count = system_info['cpu_count']
        is_containerized = system_info['is_containerized']
        
        # Determine processing quality based on available resources
        if memory_gb >= 8 and cpu_count >= 4:
            quality = "high"
            max_memory = min(memory_gb * 0.6, 6.0)  # Use up to 60% of memory, max 6GB
        elif memory_gb >= 4 and cpu_count >= 2:
            quality = "balanced"
            max_memory = min(memory_gb * 0.5, 4.0)  # Use up to 50% of memory, max 4GB
        else:
            quality = "fast"
            max_memory = min(memory_gb * 0.4, 2.0)  # Use up to 40% of memory, max 2GB
        
        # Adjust for containerized environments
        if is_containerized:
            max_memory *= 0.8  # Be more conservative in containers
        
        # Determine chunk duration based on memory
        chunk_duration = 60.0 if memory_gb < 4 else 30.0 if memory_gb < 8 else 15.0
        
        # Configure parallel processing
        max_concurrent = min(cpu_count, 4) if memory_gb >= 8 else min(cpu_count, 2)
        
        return ProcessingConfig(
            max_memory_gb=max_memory,
            processing_quality=quality,
            analysis_quality=quality,
            chunk_duration=chunk_duration,
            enable_parallel_processing=cpu_count > 1,
            max_concurrent_operations=max_concurrent,
            enable_gc_optimization=memory_gb < 8,  # More aggressive GC on lower memory systems
            cleanup_temp_files=True,
            preserve_intermediate_files=False  # Don't preserve in production
        )
    
    def _load_or_create_config(self) -> ProcessingConfig:
        """Load existing configuration or create optimal one."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Create config from loaded data
                config = ProcessingConfig(**config_data)
                return config
                
            except Exception as e:
                print(f"Warning: Failed to load config file {self.config_file}: {e}")
                print("Generating new optimal configuration...")
        
        # Generate optimal configuration
        system_info = self._detect_system_capabilities()
        config = self._get_optimal_config(system_info)
        
        # Save the generated configuration
        self.save_config(config)
        
        return config
    
    def save_config(self, config: ProcessingConfig) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Warning: Failed to save config file {self.config_file}: {e}")
            return False
    
    def get_config(self) -> ProcessingConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> ProcessingConfig:
        """Update configuration with new values."""
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        
        self.config = ProcessingConfig(**config_dict)
        self.save_config(self.config)
        
        return self.config
    
    def get_environment_specific_config(self, environment: str = "production") -> ProcessingConfig:
        """Get configuration optimized for specific environment."""
        base_config = self.config.to_dict()
        
        environment_overrides = {
            "development": {
                "preserve_intermediate_files": True,
                "cleanup_temp_files": False,
                "enable_progress_callbacks": True,
                "processing_quality": "fast"  # Faster iteration during development
            },
            "testing": {
                "max_memory_gb": 1.0,  # Conservative for CI/CD
                "processing_quality": "fast",
                "chunk_duration": 60.0,
                "enable_parallel_processing": False,
                "cleanup_temp_files": True
            },
            "production": {
                "preserve_intermediate_files": False,
                "cleanup_temp_files": True,
                "enable_gc_optimization": True,
                "enable_progress_callbacks": True
            },
            "azure_container_apps": {
                "max_memory_gb": 3.5,  # Leave room for system overhead
                "processing_quality": "balanced",
                "enable_parallel_processing": True,
                "cleanup_temp_files": True,
                "memory_safety_margin": 0.5
            },
            "azure_functions": {
                "max_memory_gb": 1.5,  # Functions have memory limits
                "processing_quality": "fast",
                "chunk_duration": 120.0,  # Larger chunks for efficiency
                "enable_parallel_processing": False,
                "cleanup_temp_files": True
            }
        }
        
        if environment in environment_overrides:
            base_config.update(environment_overrides[environment])
        
        return ProcessingConfig(**base_config)

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get or create global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_processing_config(environment: Optional[str] = None) -> ProcessingConfig:
    """Get processing configuration for specified environment."""
    manager = get_config_manager()
    
    if environment:
        return manager.get_environment_specific_config(environment)
    else:
        return manager.get_config()

def update_processing_config(**kwargs) -> ProcessingConfig:
    """Update processing configuration with new values."""
    manager = get_config_manager()
    return manager.update_config(**kwargs)

# Environment detection helper
def detect_environment() -> str:
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

if __name__ == "__main__":
    # CLI for configuration management
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "show":
            config = get_processing_config()
            print("Current Configuration:")
            print(json.dumps(config.to_dict(), indent=2))
        
        elif command == "optimize":
            environment = sys.argv[2] if len(sys.argv) > 2 else detect_environment()
            config = get_processing_config(environment)
            print(f"Optimized Configuration for {environment}:")
            print(json.dumps(config.to_dict(), indent=2))
        
        elif command == "detect":
            manager = ConfigManager()
            system_info = manager._detect_system_capabilities()
            print("System Capabilities:")
            print(json.dumps(system_info, indent=2))
            
            optimal_config = manager._get_optimal_config(system_info)
            print("\nOptimal Configuration:")
            print(json.dumps(optimal_config.to_dict(), indent=2))
        
        else:
            print("Usage: python audio_processing_config.py [show|optimize|detect] [environment]")
    else:
        print("Audio Processing Configuration Manager")
        print("Usage: python audio_processing_config.py [show|optimize|detect] [environment]")
        print("\nCommands:")
        print("  show     - Show current configuration")
        print("  optimize - Show optimized configuration for environment")
        print("  detect   - Detect system capabilities and show optimal config")
        print("\nEnvironments: development, testing, production, azure_container_apps, azure_functions")
