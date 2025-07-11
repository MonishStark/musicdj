<!-- @format -->

# Python Dependencies Documentation

## Overview

This document describes the Python dependencies for the AI Remixer application, including their purposes, versions, and security considerations.

## Dependency Files

### 1. `requirements.txt`

**Purpose**: Complete list of all Python dependencies with pinned versions
**Usage**: Primary dependency file for development and production
**Install**: `npm run python:install` or `pip install -r requirements.txt`

### 2. `requirements-dev.txt`

**Purpose**: Additional dependencies for development, testing, and debugging
**Usage**: Development environment only
**Install**: `npm run python:install-dev`

### 3. `requirements-prod.txt`

**Purpose**: Minimal set of dependencies optimized for production deployment
**Usage**: Production environments, Docker containers
**Install**: `npm run python:install-prod`

### 4. `requirements-frozen.txt`

**Purpose**: Generated file with exact versions currently installed
**Usage**: Reproducible builds, version auditing
**Generate**: `npm run python:freeze`

## Core Dependencies

### Audio Processing

- **librosa (0.11.0)**: Audio and music analysis library
- **numpy (1.23.5)**: Numerical computing foundation
- **scipy (1.15.2)**: Scientific computing algorithms
- **soundfile (0.13.1)**: Audio file I/O
- **pydub (0.25.1)**: Audio manipulation and conversion
- **audioread (3.0.1)**: Cross-library audio decoding

### Machine Learning

- **tensorflow (2.9.3)**: Deep learning framework
- **keras (2.9.0)**: High-level neural networks API
- **scikit-learn (1.6.1)**: Machine learning algorithms
- **madmom (0.16.1)**: Audio signal processing and beat detection
- **spleeter (2.4.0)**: Source separation using deep learning

### Performance & Optimization

- **numba (0.61.2)**: JIT compiler for numerical functions
- **llvmlite (0.44.0)**: LLVM bindings for Numba
- **joblib (1.4.2)**: Parallel computing and caching

## Security Considerations

### Version Pinning

All dependencies are pinned to specific versions to ensure:

- **Reproducible builds**: Same versions across environments
- **Security stability**: No automatic updates to vulnerable versions
- **Compatibility**: Tested version combinations

### Security Scanning

Regular security audits should be performed:

```bash
# Check for known vulnerabilities
npm run python:safety

# Check dependency conflicts
npm run python:check

# Audit dependencies (if available)
npm run python:audit
```

### Update Strategy

1. **Monthly Reviews**: Check for security updates
2. **Staged Updates**: Test in development before production
3. **Version Testing**: Verify compatibility after updates
4. **Security Patches**: Priority updates for critical vulnerabilities

## Environment Management

### Virtual Environment

The project uses a Python virtual environment located at `venv310/`:

- **Python Version**: 3.10.0
- **Activation**: `venv310/Scripts/activate` (Windows)
- **Deactivation**: `deactivate`

### NPM Scripts

```bash
# Install production dependencies
npm run python:install-prod

# Install all dependencies (dev + prod)
npm run python:install-dev

# Generate frozen requirements
npm run python:freeze

# Check dependency health
npm run python:check

# Security scan
npm run python:safety
```

## Dependency Categories

### Essential Runtime (Production)

- Audio processing: librosa, numpy, scipy, soundfile, pydub
- Machine learning: tensorflow, keras, scikit-learn
- Audio analysis: madmom, spleeter
- Performance: numba, joblib

### Supporting Libraries

- FFmpeg integration: ffmpeg-python
- Data processing: pandas
- Networking: requests, urllib3
- Serialization: protobuf, msgpack

### Development Only

- Testing: pytest, pytest-cov
- Code quality: black, flake8, mypy
- Documentation: sphinx
- Debugging: ipython, ipdb

## Troubleshooting

### Common Issues

1. **TensorFlow Installation**: May require specific CUDA versions
2. **Audio Libraries**: Require system audio codecs (FFmpeg)
3. **Compilation**: Some packages need C++ build tools
4. **Memory**: Large models may require sufficient RAM

### Platform-Specific Notes

- **Windows**: May need Visual Studio Build Tools
- **macOS**: May need Xcode command line tools
- **Linux**: May need development packages (python3-dev, etc.)

## Maintenance

### Regular Tasks

1. **Weekly**: Monitor for security advisories
2. **Monthly**: Review and test dependency updates
3. **Quarterly**: Full dependency audit and cleanup
4. **Annually**: Major version updates and testing

### Update Process

1. Test updates in development environment
2. Run full test suite
3. Check for breaking changes
4. Update documentation
5. Deploy to staging
6. Monitor for issues
7. Deploy to production

## Docker Considerations

For containerized deployments:

- Use `requirements-prod.txt` for smaller images
- Multi-stage builds to exclude build dependencies
- Pin base Python image version
- Use security-focused base images

## Compliance & Auditing

### License Compliance

Regular review of dependency licenses for compliance:

- Most dependencies use permissive licenses (MIT, BSD, Apache)
- TensorFlow uses Apache 2.0
- Some audio libraries may have specific terms

### Audit Trail

- All dependency changes should be documented
- Version changes tracked in git history
- Security updates prioritized and documented
- Regular compliance reports generated
