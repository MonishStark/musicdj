<!-- @format -->

# Configuration Documentation

## Overview

This document provides comprehensive documentation for all configuration files and environment variables used in the Music DJ Feature application. The application uses a modern tech stack with TypeScript, React, Node.js, and PostgreSQL.

## Environment Variables

### Required Environment Variables

The application requires the following environment variables to be configured. Use the `.env.example` file as a template:

#### Database Configuration

```bash
# PostgreSQL database connection settings
DATABASE_HOST=localhost              # Database server hostname
DATABASE_PORT=5432                  # Database server port (default: 5432)
DATABASE_USER=your_username         # Database username
DATABASE_PASSWORD=your_password     # Database password
DATABASE_NAME=music_dj_db           # Database name
DATABASE_URL=postgresql://username:password@localhost:5432/music_dj_db  # Full connection string
```

**Purpose**: These variables configure the PostgreSQL database connection for storing track metadata, processing settings, and user data.

**Usage**: Used by:

- `server/storage.ts` - Main database connection
- `drizzle.config.ts` - Database migrations configuration
- `scripts/test-db-connection.js` - Database connectivity testing
- `scripts/run-migration.js` - Database migration execution

#### Application Configuration

```bash
# Core application settings
NODE_ENV=development                # Environment mode (development/production)
PORT=5000                          # Server port (default: 5000)
UPLOADS_DIR=./uploads              # Directory for uploaded audio files
RESULTS_DIR=./results              # Directory for processed audio files
MAX_FILE_SIZE_MB=15                # Maximum upload file size in MB
```

**Purpose**: These variables control the application's runtime behavior and file storage locations.

**Usage**: Used by:

- `server/routes.ts` - File upload and storage paths
- `server/index.ts` - Server port configuration
- File upload middleware for size limits

#### Python Configuration

```bash
# Python environment for audio processing
PYTHON_PATH=python                 # Python executable path
```

**Purpose**: Configures the Python environment used for audio processing tasks like source separation and beat detection.

**Usage**: Used by Python shell integration for audio processing algorithms.

### Optional Environment Variables

#### Development Environment

```bash
REPL_ID=undefined                  # Replit environment identifier (auto-set in Replit)
```

**Purpose**: Used for Replit-specific development features and plugins.

**Usage**: Used in `vite.config.ts` for conditional plugin loading.

## Configuration Files

### 1. package.json

**Location**: `/package.json`

**Purpose**: Defines project metadata, dependencies, and npm scripts for the Node.js application.

#### Key Scripts

```json
{
	"scripts": {
		"dev": "cross-env NODE_ENV=development tsx ./server/index.ts",
		"build": "vite build && esbuild ./server/index.ts --platform=node --packages=external --bundle --format=esm --outdir=dist",
		"start": "NODE_ENV=production node dist/index.js",
		"check": "tsc",
		"db:push": "drizzle-kit push",
		"setup": "npm install && mkdir -p uploads results",
		"clean": "rimraf dist uploads/* results/*",
		"db:test": "node scripts/test-db-connection.js"
	}
}
```

#### Critical Dependencies

- **Frontend**: React 18, TanStack Query, Framer Motion, Radix UI components
- **Backend**: Express.js, Drizzle ORM, PostgreSQL, Multer for file uploads
- **Audio Processing**: Python Shell integration for audio analysis
- **Build Tools**: Vite, TypeScript, Tailwind CSS

### 2. vite.config.ts

**Location**: `/vite.config.ts`

**Purpose**: Configures Vite build tool for frontend development and production builds.

#### Key Configuration

```typescript
export default defineConfig({
	plugins: [
		react(), // React support
		runtimeErrorOverlay(), // Development error overlay
	],
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "client", "src"), // Client source alias
			"@shared": path.resolve(__dirname, "shared"), // Shared types alias
			"@assets": path.resolve(__dirname, "attached_assets"), // Assets alias
		},
	},
	root: path.resolve(__dirname, "client"), // Frontend root directory
	build: {
		outDir: path.resolve(__dirname, "dist/public"), // Build output directory
		emptyOutDir: true,
	},
});
```

**Features**:

- Path aliases for clean imports
- React plugin integration
- Development error handling
- Replit-specific plugins in development

### 3. tailwind.config.ts

**Location**: `/tailwind.config.ts`

**Purpose**: Configures Tailwind CSS framework with custom theme, colors, and animations.

#### Key Features

- **Dark Mode**: Class-based dark mode support
- **Custom Colors**: Primary, secondary, accent color schemes with CSS variables
- **Animations**: Custom accordion animations and Tailwind animate plugin
- **Typography**: Enhanced typography plugin for rich text content
- **Sidebar Components**: Dedicated color scheme for sidebar elements

#### Content Sources

```typescript
content: ["./client/index.html", "./client/src/**/*.{js,jsx,ts,tsx}"];
```

### 4. tsconfig.json

**Location**: `/tsconfig.json`

**Purpose**: TypeScript compiler configuration for the entire project.

#### Key Configuration

```json
{
	"include": ["client/src/**/*", "shared/**/*", "server/**/*"],
	"exclude": ["node_modules", "build", "dist", "**/*.test.ts"],
	"compilerOptions": {
		"module": "ESNext",
		"strict": true,
		"jsx": "preserve",
		"baseUrl": ".",
		"paths": {
			"@/*": ["./client/src/*"],
			"@shared/*": ["./shared/*"]
		}
	}
}
```

**Features**:

- Strict type checking enabled
- Path mapping for aliases
- Modern ESNext module system
- DOM and Node.js type support

### 5. drizzle.config.ts

**Location**: `/drizzle.config.ts`

**Purpose**: Configures Drizzle ORM for database schema management and migrations.

```typescript
export default defineConfig({
	out: "./migrations", // Migration files output directory
	schema: "./shared/schema.ts", // Database schema definition
	dialect: "postgresql", // Database dialect
	dbCredentials: {
		host: process.env.DATABASE_HOST!,
		port: parseInt(process.env.DATABASE_PORT || "5432"),
		user: process.env.DATABASE_USER!,
		password: process.env.DATABASE_PASSWORD!,
		database: process.env.DATABASE_NAME!,
	},
});
```

**Usage**: Used by Drizzle Kit for:

- Generating database migrations
- Pushing schema changes to database
- Type-safe database operations

### 6. components.json

**Location**: `/components.json`

**Purpose**: Configures shadcn/ui component library integration.

#### Configuration

```json
{
	"style": "new-york", // Component style variant
	"rsc": false, // Not using React Server Components
	"tsx": true, // TypeScript JSX support
	"tailwind": {
		"config": "tailwind.config.ts",
		"css": "client/src/index.css",
		"baseColor": "neutral",
		"cssVariables": true
	},
	"aliases": {
		"components": "@/components",
		"utils": "@/lib/utils",
		"ui": "@/components/ui"
	}
}
```

**Purpose**: Ensures consistent component generation and path resolution for UI components.

### 7. postcss.config.js

**Location**: `/postcss.config.js`

**Purpose**: Configures PostCSS for CSS processing with Tailwind CSS and Autoprefixer.

```javascript
export default {
	plugins: {
		tailwindcss: {}, // Tailwind CSS processing
		autoprefixer: {}, // Vendor prefix automation
	},
};
```

### 8. pyproject.toml

**Location**: `/pyproject.toml`

**Purpose**: Python project configuration for audio processing dependencies.

**Note**: Currently minimal configuration. In production, this would contain Python dependencies for:

- Audio processing libraries (librosa, spleeter)
- Machine learning frameworks (TensorFlow, PyTorch)
- Audio format support (pydub, soundfile)

## Configuration Usage Patterns

### Environment Variable Loading

The application loads environment variables using `dotenv` in the server entry point:

```typescript
// server/index.ts
import "dotenv/config"; // Loads .env file automatically
```

### Database Connection

Environment variables are used consistently across database-related files:

```typescript
// server/storage.ts
if (!process.env.DATABASE_URL) {
	throw new Error("DATABASE_URL environment variable is required");
}

const db = drizzle(postgres(process.env.DATABASE_URL));
```

### File Storage Configuration

Directory paths are configurable via environment variables with sensible defaults:

```typescript
// server/routes.ts
const UPLOADS_DIR =
	process.env.UPLOADS_DIR || path.join(process.cwd(), "uploads");
const RESULTS_DIR =
	process.env.RESULTS_DIR || path.join(process.cwd(), "results");
```

## Setup Instructions

### 1. Environment Setup

1. Copy `.env.example` to `.env`
2. Configure database credentials
3. Set appropriate directory paths
4. Configure Python environment if needed

### 2. Database Setup

```bash
# Install dependencies
npm install

# Push database schema
npm run db:push

# Test database connection
npm run db:test
```

### 3. Development Setup

```bash
# Create necessary directories and install deps
npm run setup

# Start development server
npm run dev
```

### 4. Production Setup

```bash
# Build application
npm run build

# Start production server
npm start
```

## Security Considerations

### Environment Variables

- **Never commit** `.env` files to version control
- Use strong database passwords
- Restrict database access to application server only
- Use environment-specific configurations for different deployment stages

### File Storage

- Validate file types and sizes on upload
- Implement proper file permissions on storage directories
- Consider using cloud storage for production deployments
- Regularly clean up temporary files

### Database Security

- Use connection pooling for production
- Implement proper backup strategies
- Use SSL connections for remote databases
- Follow principle of least privilege for database users

## Troubleshooting

### Common Configuration Issues

1. **Database Connection Failures**

   - Verify DATABASE_URL format
   - Check database server availability
   - Confirm credentials and permissions

2. **File Upload Issues**

   - Ensure UPLOADS_DIR exists and is writable
   - Check MAX_FILE_SIZE_MB setting
   - Verify disk space availability

3. **Build Failures**

   - Check TypeScript path aliases in tsconfig.json
   - Verify all dependencies are installed
   - Ensure environment variables are set

4. **Python Integration Issues**
   - Verify PYTHON_PATH points to correct executable
   - Check Python dependencies are installed
   - Ensure Python environment is activated

This configuration documentation should be updated whenever new environment variables or configuration files are added to the project.
