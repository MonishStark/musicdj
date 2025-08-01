<!-- @format -->

# 🎵 Music DJ Feature - Advanced Audio Processing Platform

A comprehensive web application that empowers DJs and music producers to create extended versions of tracks with custom intros and outros. This platform features AI-powered audio processing, background job queues, real-time progress tracking, and memory-optimized processing capabilities.

## ✨ Key Features

### 🎧 Audio Processing

- **Multi-format Support**: Upload MP3, WAV, FLAC, AIFF audio files
- **AI-Powered Separation**: Advanced audio separation into components (vocals, drums, bass, other)
- **Smart Beat Detection**: Automatic tempo analysis and beat detection
- **Custom Extensions**: Configurable intro and outro lengths
- **Memory Optimization**: 60-80% memory reduction with optimized processing
- **Multiple Versions**: Support for multiple processed versions per track

### 🚀 Performance & Scalability

- **Background Job Queue**: Asynchronous processing with Redis/Bull queue system
- **Real-time Progress**: Live processing updates via WebSocket
- **Priority Processing**: 4-level priority system (Low, Normal, High, Critical)
- **Streaming Uploads**: Memory-efficient upload for large files (up to 500MB)
- **Auto-retry Logic**: Exponential backoff retry mechanism for failed jobs
- **Health Monitoring**: Comprehensive queue health checks and metrics

### 💻 User Experience

- **Dual Upload Modes**: Standard upload (≤15MB) and streaming upload (≤500MB)
- **Real-time Preview**: Audio preview capabilities
- **Progress Tracking**: Visual progress indicators with detailed status
- **Job Management**: Cancel, pause, and monitor processing jobs
- **Admin Dashboard**: Advanced monitoring interface for administrators
- **Responsive Design**: Modern UI with Tailwind CSS and shadcn/ui components

## 📋 Prerequisites

### Required Software

- **Node.js**: v18 or later ([Download](https://nodejs.org/))
- **Python**: 3.8 or later ([Download](https://www.python.org/downloads/))
- **FFmpeg**: For audio processing ([Download](https://ffmpeg.org/download.html))
- **Redis**: For job queue system ([Installation Guide](#redis-setup))
- **PostgreSQL**: For database ([Download](https://www.postgresql.org/download/))

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+ for large files
- **Storage**: At least 2GB free space for uploads and processing
- **CPU**: Multi-core processor recommended for faster processing

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd music-dj-feature-new-code

# Install dependencies and setup directories
npm run setup

# Install additional job queue dependencies
npm install bull ioredis uuid socket.io socket.io-client @bull-board/express @bull-board/ui
npm install --save-dev @types/bull @types/uuid
```

### 2. Python Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv310

# Activate virtual environment
# Windows:
venv310\Scripts\activate
# macOS/Linux:
source venv310/bin/activate

# Install Python dependencies
pip install librosa numpy pydub madmom spleeter tensorflow torch
pip install python-shell  # For Node.js-Python integration
```

### 3. Database Setup

```bash
# Configure your PostgreSQL database
# Update DATABASE_URL in .env file

# Push database schema
npm run db:push

# Test database connection
npm run db:test
```

### 4. Redis Setup

Choose one of the following options:

#### Option A: Local Redis (Development)

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install redis-server
sudo systemctl start redis-server

# macOS with Homebrew
brew install redis
brew services start redis

# Windows (using Chocolatey)
choco install redis-64
redis-server
```

#### Option B: Docker Redis (Recommended)

```bash
# Start Redis container
docker run -d --name redis-dj -p 6379:6379 redis:7-alpine

# Verify Redis is running
docker exec redis-dj redis-cli ping
```

#### Option C: Cloud Redis (Production)

- [Redis Cloud](https://redis.com/redis-enterprise-cloud/)
- [AWS ElastiCache](https://aws.amazon.com/elasticache/)
- [Azure Cache for Redis](https://azure.microsoft.com/services/cache/)

### 5. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/musicdj"

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=           # Optional for local development
REDIS_DB=0

# Server Configuration
NODE_ENV=development
CLIENT_URL=http://localhost:5000

# File Storage
UPLOADS_DIR=./uploads
RESULTS_DIR=./results
MAX_FILE_SIZE=500000000   # 500MB

# Processing Configuration
PYTHON_PATH=python        # Or full path to Python executable
ENABLE_MEMORY_OPTIMIZATION=true
DEFAULT_INTRO_LENGTH=16
DEFAULT_OUTRO_LENGTH=16

# Security (generate secure values for production)
JWT_SECRET=your-jwt-secret-here
SESSION_SECRET=your-session-secret-here
```

### 6. Start the Application

```bash
# Development mode (with hot reload)
npm run dev

# The application will be available at:
# Frontend: http://localhost:5000
# Backend API: http://localhost:5000/api
# Job Queue Dashboard: http://localhost:5000/admin/queues (if enabled)
```

## 📁 Project Structure

```
music-dj-feature-new-code/
├── client/                     # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── EnhancedUpload.tsx
│   │   │   ├── JobQueueMonitor.tsx
│   │   │   ├── EnhancedSettingsPanel.tsx
│   │   │   └── ...
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utility functions
│   │   └── pages/              # Page components
│   └── index.html
├── server/                     # Express backend
│   ├── index.ts                # Main server file
│   ├── routes.ts               # API routes
│   ├── storage.ts              # Database operations
│   ├── jobQueue.ts             # Job queue system (Redis)
│   ├── jobQueueSimple.ts       # Fallback job queue (no Redis)
│   ├── websocketManager.ts     # WebSocket management
│   ├── jobQueueRoutes.ts       # Job queue API routes
│   ├── audioProcessor.py       # Python audio processing
│   └── audioProcessor_optimized.py # Memory-optimized processing
├── shared/                     # Shared TypeScript types
│   └── schema.ts
├── pretrained_models/          # AI models for audio processing
├── uploads/                    # Uploaded audio files
├── results/                    # Processed audio files
├── scripts/                    # Utility scripts
└── documentation/
    └── JOB_QUEUE_IMPLEMENTATION.md
```

## 🔧 Development Workflow

### Running in Development Mode

```bash
# Start with hot reload
npm run dev

# Run with specific environment
NODE_ENV=development npm run dev

# Check TypeScript types
npm run check

# Test database connection
npm run db:test
```

### Building for Production

```bash
# Build frontend and backend
npm run build

# Start production server
npm start

# Clean build artifacts
npm run clean
```

### Testing

```bash
# Test streaming upload functionality
npm run test:streaming

# Start with batch file (Windows)
npm run start:dev
```

## 🎛️ Usage Guide

### 1. Upload Audio Files

#### Standard Upload (≤15MB)

- Fast upload for most audio files
- Immediate processing start
- Suitable for typical song files

#### Streaming Upload (≤500MB)

- Memory-efficient for large files
- Real-time progress tracking
- Chunked upload with resume capability

### 2. Configure Processing Settings

#### Basic Settings

- **Intro Length**: Duration for intro extension (default: 16 seconds)
- **Outro Length**: Duration for outro extension (default: 16 seconds)
- **Preserve Vocals**: Keep vocal components in extensions
- **Beat Detection**: Auto, manual, or custom BPM

#### Advanced Settings

- **Priority Level**: Low, Normal, High, or Critical processing priority
- **Memory Optimization**: Enable 60-80% memory reduction
- **Processing Mode**: Standard or optimized Python processing
- **Queue Settings**: Background or immediate processing

### 3. Monitor Processing

#### Real-time Updates

- Live progress percentage
- Current processing stage
- Estimated time remaining
- Memory usage monitoring

#### Job Management

- View active, completed, and failed jobs
- Cancel pending or active jobs
- Retry failed jobs with different settings
- Download completed processed files

### 4. Admin Features

#### Queue Management

- Monitor queue statistics
- Pause/resume job processing
- Bulk processing capabilities
- Health monitoring and alerts

#### System Monitoring

- Redis connection status
- Processing performance metrics
- Error rates and failure analysis
- Resource usage tracking

## 🐛 Troubleshooting

### Common Issues

#### Redis Connection Failed

```bash
Error: Redis connection failed
```

**Solutions:**

1. Check Redis server status: `redis-cli ping`
2. Verify Redis configuration in `.env`
3. Check firewall settings for port 6379
4. For development, the app will fallback to direct processing mode

#### Python Dependencies Missing

```bash
ModuleNotFoundError: No module named 'librosa'
```

**Solutions:**

1. Activate virtual environment: `venv310\Scripts\activate`
2. Install missing packages: `pip install librosa numpy pydub madmom spleeter`
3. Verify Python path in `.env`: `PYTHON_PATH=python`

#### Memory Issues During Processing

```bash
MemoryError: Unable to allocate array
```

**Solutions:**

1. Enable memory optimization: Set `ENABLE_MEMORY_OPTIMIZATION=true`
2. Use optimized processing mode in settings
3. Process smaller files or reduce concurrency
4. Increase system RAM or use cloud processing

#### File Upload Errors

```bash
Error: File too large
```

**Solutions:**

1. Use streaming upload for files >15MB
2. Check `MAX_FILE_SIZE` in `.env`
3. Verify supported file formats: MP3, WAV, FLAC, AIFF
4. Ensure sufficient disk space

### Debug Commands

#### Check System Status

```bash
# Application health
curl http://localhost:5000/api/health/job-queue

# Queue statistics
curl http://localhost:5000/api/admin/queue-stats

# Redis status
redis-cli info memory
```

#### Monitor Logs

```bash
# Application logs (if configured)
tail -f logs/app.log

# Redis logs
tail -f /var/log/redis/redis-server.log

# System resources
# Windows:
Get-Process | Where-Object {$_.ProcessName -eq "node" -or $_.ProcessName -eq "python"}

# macOS/Linux:
ps aux | grep -E "(node|python)"
```

## 🚀 Deployment

### Environment Preparation

#### Production Environment Variables

```bash
NODE_ENV=production
DATABASE_URL="postgresql://user:pass@prod-host:5432/musicdj"
REDIS_HOST=your-redis-cluster-endpoint
REDIS_PASSWORD=secure-redis-password
CLIENT_URL=https://your-domain.com
ENABLE_BULL_BOARD=true
BULL_BOARD_USERNAME=admin
BULL_BOARD_PASSWORD=secure-admin-password
```

### Docker Deployment

#### Docker Compose Setup

```yaml
# docker-compose.yml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://user:pass@postgres:5432/musicdj
    depends_on:
      - redis
      - postgres
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: musicdj
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  redis_data:
  postgres_data:
```

#### Build and Deploy

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale processing workers
docker-compose up -d --scale app=3
```

### Cloud Deployment Options

#### Vercel (Frontend + Serverless Functions)

- Deploy React frontend to Vercel
- Use Vercel Functions for API endpoints
- Connect to external Redis and PostgreSQL

#### Railway (Full Stack)

- Deploy entire application to Railway
- Auto-provisioned PostgreSQL and Redis
- Easy environment variable management

#### AWS/Azure/GCP

- Deploy using Docker containers
- Use managed Redis (ElastiCache/Azure Cache)
- Use managed PostgreSQL (RDS/Azure Database)

## 📈 Performance Optimization

### Memory Optimization

- Enable optimized Python processing: 60-80% memory reduction
- Use streaming uploads for large files
- Configure appropriate queue concurrency limits
- Monitor memory usage through admin dashboard

### Processing Speed

- Utilize priority queues for urgent processing
- Enable background job processing
- Configure multiple processing workers
- Use SSD storage for faster I/O operations

### Scalability

- Horizontal scaling with multiple app instances
- Redis cluster for queue distribution
- Load balancing for high traffic
- CDN for static asset delivery

## 🤝 Contributing

### Development Setup

1. Follow the setup instructions above
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and test thoroughly
4. Update documentation if needed
5. Submit a pull request with detailed description

### Code Standards

- Use TypeScript for type safety
- Follow ESLint and Prettier configurations
- Write comprehensive tests for new features
- Document complex functions and modules
- Follow conventional commit messages

### Testing Guidelines

- Test file upload functionality
- Verify audio processing pipeline
- Test job queue operations
- Validate error handling
- Performance testing for large files

## 📝 API Documentation

### Core Endpoints

#### Upload Management

- `POST /api/upload` - Standard file upload
- `POST /api/upload/streaming/start` - Start streaming upload
- `POST /api/upload/streaming/chunk` - Upload file chunk
- `POST /api/upload/streaming/complete` - Complete streaming upload

#### Job Queue Management

- `POST /api/tracks/:id/process-async` - Start async processing
- `GET /api/jobs/:jobId/status` - Get job status
- `DELETE /api/jobs/:jobId` - Cancel job
- `POST /api/tracks/process-bulk` - Bulk processing

#### Admin Operations

- `GET /api/admin/queue-stats` - Queue statistics
- `POST /api/admin/queue-control` - Control queue operations
- `GET /api/health/job-queue` - System health check

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For technical support, feature requests, or bug reports:

1. Check the [troubleshooting section](#-troubleshooting) above
2. Review the [comprehensive documentation](documentation/JOB_QUEUE_IMPLEMENTATION.md)
3. Create an issue in the repository with detailed information
4. Include system information and error logs when reporting bugs

---

**🎵 Happy DJ-ing with enhanced audio processing capabilities! 🎵**
