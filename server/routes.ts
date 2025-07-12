/** @format */

import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import {
	processingSettingsSchema,
	updateAudioTrackSchema,
} from "@shared/schema";
import multer from "multer";
import path from "path";
import fs from "fs";
import { PythonShell } from "python-shell";

// Audit logging configuration
const AUDIT_CONFIG = {
	ENABLED: process.env.AUDIT_LOGGING_ENABLED !== "false",
	LOG_TO_FILE: process.env.AUDIT_LOG_TO_FILE === "true",
	LOG_FILE_PATH:
		process.env.AUDIT_LOG_FILE || path.join(process.cwd(), "logs", "audit.log"),
	LOG_TO_CONSOLE:
		process.env.NODE_ENV === "development" ||
		process.env.AUDIT_LOG_TO_CONSOLE === "true",
	INCLUDE_REQUEST_DETAILS: process.env.AUDIT_INCLUDE_REQUEST_DETAILS === "true",
	RETENTION_DAYS: parseInt(process.env.AUDIT_LOG_RETENTION_DAYS || "90", 10),
};

// Audit event types
enum AuditEventType {
	// Authentication & Authorization
	AUTH_SUCCESS = "AUTH_SUCCESS",
	AUTH_FAILURE = "AUTH_FAILURE",
	UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS",

	// File Operations
	FILE_UPLOAD = "FILE_UPLOAD",
	FILE_UPLOAD_REJECTED = "FILE_UPLOAD_REJECTED",
	FILE_ACCESS = "FILE_ACCESS",
	FILE_ACCESS_DENIED = "FILE_ACCESS_DENIED",
	FILE_DOWNLOAD = "FILE_DOWNLOAD",
	FILE_DELETE = "FILE_DELETE",

	// Processing Operations
	AUDIO_PROCESSING_START = "AUDIO_PROCESSING_START",
	AUDIO_PROCESSING_SUCCESS = "AUDIO_PROCESSING_SUCCESS",
	AUDIO_PROCESSING_FAILURE = "AUDIO_PROCESSING_FAILURE",

	// Security Events
	PATH_TRAVERSAL_ATTEMPT = "PATH_TRAVERSAL_ATTEMPT",
	INVALID_FILE_TYPE = "INVALID_FILE_TYPE",
	FILE_SIZE_VIOLATION = "FILE_SIZE_VIOLATION",
	MALICIOUS_FILENAME = "MALICIOUS_FILENAME",
	RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED",
	SUSPICIOUS_REQUEST = "SUSPICIOUS_REQUEST",

	// System Events
	SYSTEM_ERROR = "SYSTEM_ERROR",
	CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE",
	SERVICE_START = "SERVICE_START",
	SERVICE_STOP = "SERVICE_STOP",
}

// Audit event severity levels
enum AuditSeverity {
	LOW = "LOW",
	MEDIUM = "MEDIUM",
	HIGH = "HIGH",
	CRITICAL = "CRITICAL",
}

interface AuditEventDetails {
	// File operation details
	fileName?: string;
	filePath?: string;
	fileSize?: number;
	fileType?: string;
	uploadId?: string;

	// Processing details
	processingTime?: number;
	algorithm?: string;
	settings?: ProcessingSettings;
	errorMessage?: string;

	// Security details
	pathTraversalAttempt?: string;
	maliciousPattern?: string;
	validationError?: string;

	// Request details
	endpoint?: string;
	method?: string;
	statusCode?: number;
	responseTime?: number;

	// Additional context (for any truly unknown properties)
	[key: string]: unknown;
}

interface ProcessingSettings {
	introLength: number;
	outroLength: number;
	preserveVocals: boolean;
	beatDetection: string;
	algorithm?: string;
	tempo?: number;
	key?: string;
}

interface AuditEvent {
	timestamp: string;
	eventType: AuditEventType;
	severity: AuditSeverity;
	userId?: string | number;
	sessionId?: string;
	ipAddress?: string;
	userAgent?: string;
	resource?: string;
	action?: string;
	outcome: "SUCCESS" | "FAILURE";
	details: AuditEventDetails;
	requestId?: string;
}

/**
 * Comprehensive audit logging system for security events
 */
class AuditLogger {
	private static instance: AuditLogger;
	private logDir: string;

	private constructor() {
		this.logDir = path.dirname(AUDIT_CONFIG.LOG_FILE_PATH);
		this.ensureLogDirectoryExists();
	}

	public static getInstance(): AuditLogger {
		if (!AuditLogger.instance) {
			AuditLogger.instance = new AuditLogger();
		}
		return AuditLogger.instance;
	}

	private ensureLogDirectoryExists(): void {
		try {
			if (!fs.existsSync(this.logDir)) {
				fs.mkdirSync(this.logDir, { recursive: true, mode: 0o750 });
			}
		} catch (error) {
			console.error("Failed to create audit log directory:", error);
		}
	}

	private extractClientInfo(req?: Request): Partial<AuditEvent> {
		if (!req) return {};

		return {
			ipAddress: this.getClientIpAddress(req),
			userAgent: req.headers["user-agent"],
			requestId: this.generateRequestId(),
		};
	}

	private getClientIpAddress(req: Request): string {
		return (
			(req.headers["x-forwarded-for"] as string)?.split(",")[0] ||
			(req.headers["x-real-ip"] as string) ||
			req.socket.remoteAddress ||
			"unknown"
		);
	}

	private generateRequestId(): string {
		return Date.now().toString(36) + Math.random().toString(36).substr(2);
	}

	private sanitizeForLogging(
		data: Record<string, unknown>
	): Record<string, unknown> {
		const sanitized = { ...data };

		// Remove or mask sensitive information
		const sensitiveFields = ["password", "token", "secret", "key", "auth"];
		for (const field of sensitiveFields) {
			if (sanitized[field]) {
				sanitized[field] = "[REDACTED]";
			}
		}

		// Limit file path exposure
		if (sanitized.filePath && typeof sanitized.filePath === "string") {
			sanitized.filePath = path.basename(sanitized.filePath);
		}

		// Truncate long strings
		Object.keys(sanitized).forEach((key) => {
			const value = sanitized[key];
			if (typeof value === "string" && value.length > 500) {
				sanitized[key] = value.substring(0, 500) + "...[TRUNCATED]";
			}
		});

		return sanitized;
	}

	private async writeToFile(auditEvent: AuditEvent): Promise<void> {
		if (!AUDIT_CONFIG.LOG_TO_FILE) return;

		try {
			const logEntry = JSON.stringify(auditEvent) + "\n";
			await fs.promises.appendFile(AUDIT_CONFIG.LOG_FILE_PATH, logEntry, {
				mode: 0o640,
			});
		} catch (error) {
			console.error("Failed to write to audit log file:", error);
		}
	}

	private logToConsole(auditEvent: AuditEvent): void {
		if (!AUDIT_CONFIG.LOG_TO_CONSOLE) return;

		const logLevel =
			auditEvent.severity === AuditSeverity.CRITICAL ||
			auditEvent.severity === AuditSeverity.HIGH
				? "error"
				: "info";

		console[logLevel](
			`[AUDIT] ${auditEvent.eventType} - ${auditEvent.outcome}`,
			{
				timestamp: auditEvent.timestamp,
				severity: auditEvent.severity,
				userId: auditEvent.userId,
				ipAddress: auditEvent.ipAddress,
				resource: auditEvent.resource,
				details: auditEvent.details,
			}
		);
	}

	public async log(
		eventType: AuditEventType,
		severity: AuditSeverity,
		outcome: "SUCCESS" | "FAILURE",
		details: AuditEventDetails = {},
		req?: Request,
		userId?: string | number
	): Promise<void> {
		if (!AUDIT_CONFIG.ENABLED) return;

		const auditEvent: AuditEvent = {
			timestamp: new Date().toISOString(),
			eventType,
			severity,
			outcome,
			details: this.sanitizeForLogging(details),
			userId: userId || demoUser.id,
			...this.extractClientInfo(req),
		};

		// Add request details if configured
		if (AUDIT_CONFIG.INCLUDE_REQUEST_DETAILS && req) {
			auditEvent.resource = req.originalUrl || req.url;
			auditEvent.action = req.method;
		}

		// Log to console
		this.logToConsole(auditEvent);

		// Log to file
		await this.writeToFile(auditEvent);
	}

	// Convenience methods for common security events
	public async logFileUpload(
		req: Request,
		filename: string,
		fileSize: number,
		outcome: "SUCCESS" | "FAILURE",
		reason?: string
	): Promise<void> {
		await this.log(
			AuditEventType.FILE_UPLOAD,
			outcome === "SUCCESS" ? AuditSeverity.LOW : AuditSeverity.MEDIUM,
			outcome,
			{ filename, fileSize, reason },
			req
		);
	}

	public async logFileAccess(
		req: Request,
		filePath: string,
		outcome: "SUCCESS" | "FAILURE",
		reason?: string
	): Promise<void> {
		await this.log(
			AuditEventType.FILE_ACCESS,
			outcome === "SUCCESS" ? AuditSeverity.LOW : AuditSeverity.MEDIUM,
			outcome,
			{ filePath: path.basename(filePath), reason },
			req
		);
	}

	public async logSecurityViolation(
		req: Request,
		violationType: AuditEventType,
		details: AuditEventDetails
	): Promise<void> {
		await this.log(violationType, AuditSeverity.HIGH, "FAILURE", details, req);
	}

	public async logProcessingEvent(
		trackId: number,
		eventType: AuditEventType,
		outcome: "SUCCESS" | "FAILURE",
		details: AuditEventDetails = {}
	): Promise<void> {
		await this.log(
			eventType,
			outcome === "SUCCESS" ? AuditSeverity.LOW : AuditSeverity.MEDIUM,
			outcome,
			{ trackId, ...details }
		);
	}

	public async logSystemEvent(
		eventType: AuditEventType,
		details: AuditEventDetails = {}
	): Promise<void> {
		await this.log(eventType, AuditSeverity.MEDIUM, "SUCCESS", details);
	}

	// Log cleanup method (should be called periodically)
	public async cleanupOldLogs(): Promise<void> {
		if (!AUDIT_CONFIG.LOG_TO_FILE) return;

		try {
			const stats = await fs.promises.stat(AUDIT_CONFIG.LOG_FILE_PATH);
			const fileAge = Date.now() - stats.mtime.getTime();
			const maxAge = AUDIT_CONFIG.RETENTION_DAYS * 24 * 60 * 60 * 1000;

			if (fileAge > maxAge) {
				const backupPath = `${AUDIT_CONFIG.LOG_FILE_PATH}.${Date.now()}.old`;
				await fs.promises.rename(AUDIT_CONFIG.LOG_FILE_PATH, backupPath);
				
			}
		} catch (error) {
			console.error("Failed to cleanup old audit logs:", error);
		}
	}
}

// Global audit logger instance
const auditLogger = AuditLogger.getInstance();

// Setup multer for file uploads
const uploadsDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadsDir)) {
	fs.mkdirSync(uploadsDir);
}

const resultDir = path.join(process.cwd(), "results");
if (!fs.existsSync(resultDir)) {
	fs.mkdirSync(resultDir);
}

// Security: Define allowed directories for file operations
const ALLOWED_DIRECTORIES = [path.resolve(uploadsDir), path.resolve(resultDir)];

// Security: Allowed file extensions
const ALLOWED_EXTENSIONS = [".mp3", ".wav", ".flac", ".aiff"];

// Security: Maximum file size from environment or default to 15MB
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE || "15728640", 10);

// Security: Maximum processing time from environment or default to 10 minutes
const MAX_PROCESSING_TIME = parseInt(
	process.env.MAX_PROCESSING_TIME || "600000",
	10
);

// Demo user for this application (in production, implement proper user authentication)
const demoUser = { id: 1, username: "demo" };

/**
 * Security function to validate and sanitize file paths
 * @param filePath - The file path to validate
 * @returns Sanitized path if valid, null if invalid
 */
function validateAndSanitizePath(filePath: string): string | null {
	try {
		// Resolve the absolute path to prevent traversal
		const resolvedPath = path.resolve(filePath);

		// Check if the resolved path is within allowed directories
		const isInAllowedDirectory = ALLOWED_DIRECTORIES.some(
			(allowedDir) =>
				resolvedPath.startsWith(allowedDir + path.sep) ||
				resolvedPath === allowedDir
		);

		if (!isInAllowedDirectory) {
			return null;
		}

		// Check file extension
		const fileExtension = path.extname(resolvedPath).toLowerCase();
		if (!ALLOWED_EXTENSIONS.includes(fileExtension)) {
			return null;
		}

		// Additional security checks
		if (resolvedPath.includes("..") || resolvedPath.includes("~")) {
			return null;
		}

		return resolvedPath;
	} catch (error) {
		return null;
	}
}

/**
 * Security function to create safe output paths
 * @param baseDir - Base directory (must be in allowed directories)
 * @param filename - Filename to create
 * @returns Safe path or null if invalid
 */
function createSafeOutputPath(
	baseDir: string,
	filename: string
): string | null {
	try {
		// Sanitize filename - remove any path separators and dangerous characters
		const sanitizedFilename = filename.replace(/[\/\\:*?"<>|]/g, "_");

		// Ensure baseDir is allowed
		const resolvedBaseDir = path.resolve(baseDir);
		const isAllowedBaseDir = ALLOWED_DIRECTORIES.some((allowedDir) =>
			resolvedBaseDir.startsWith(allowedDir)
		);

		if (!isAllowedBaseDir) {
			return null;
		}

		// Create the output path
		const outputPath = path.join(resolvedBaseDir, sanitizedFilename);

		// Validate the final path
		return validateAndSanitizePath(outputPath);
	} catch (error) {
		return null;
	}
}

/**
 * Security function to validate filename
 * @param filename - Filename to validate
 * @returns true if valid, false otherwise
 */
function isValidFilename(filename: string): boolean {
	// Check for dangerous patterns
	const dangerousPatterns = [
		/\.\./, // Path traversal
		/[<>:"|?*]/, // Invalid Windows chars
		/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/i, // Reserved Windows names
		/^\.+$/, // Only dots
		/\0/, // Null bytes
	];

	return !dangerousPatterns.some((pattern) => pattern.test(filename));
}

/**
 * Security function to safely delete files
 * @param filePath - Path to file to delete
 * @returns true if deleted successfully, false otherwise
 */
function safeDeleteFile(filePath: string): boolean {
	try {
		const safePath = validateAndSanitizePath(filePath);
		if (!safePath) {
			return false;
		}

		if (fs.existsSync(safePath)) {
			fs.unlinkSync(safePath);
			return true;
		}
		return false;
	} catch (error) {
		console.error("Error deleting file:", error);
		return false;
	}
}

const storage_config = multer.diskStorage({
	destination: function (req, file, cb) {
		cb(null, uploadsDir);
	},
	filename: function (req, file, cb) {
		const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
		const sanitizedName = file.originalname.replace(/[^a-zA-Z0-9.-]/g, "_");
		cb(null, uniqueSuffix + "-" + sanitizedName);
	},
});

const upload = multer({
	storage: storage_config,
	limits: {
		fileSize: MAX_FILE_SIZE,
	},
	fileFilter: (req, file, cb) => {
		const allowedMimeTypes = [
			"audio/mpeg",
			"audio/wav",
			"audio/flac",
			"audio/aiff",
			"audio/x-aiff",
		];

		// Validate MIME type
		if (!allowedMimeTypes.includes(file.mimetype)) {
			return cb(
				new Error(
					"Invalid file type. Only MP3, WAV, FLAC, and AIFF files are allowed."
				)
			);
		}

		// Validate filename
		if (!isValidFilename(file.originalname)) {
			return cb(new Error("Invalid filename. Please use a safer filename."));
		}

		cb(null, true);
	},
});

/**
 * Validate and sanitize processing settings
 * @param settings - Processing settings to validate
 * @returns Validated settings or null if invalid
 */
function validateProcessingSettings(
	settings: unknown
): ProcessingSettings | null {
	try {
		// Type guard to ensure settings is an object
		if (!settings || typeof settings !== "object") {
			return null;
		}

		const settingsObj = settings as Record<string, unknown>;

		// Validate intro length (1-64 bars)
		const introLength = parseInt(String(settingsObj.introLength), 10);
		if (isNaN(introLength) || introLength < 1 || introLength > 64) {
			return null;
		}

		// Validate outro length (1-64 bars)
		const outroLength = parseInt(String(settingsObj.outroLength), 10);
		if (isNaN(outroLength) || outroLength < 1 || outroLength > 64) {
			return null;
		}

		// Validate preserve vocals boolean
		const preserveVocals = Boolean(settingsObj.preserveVocals);

		// Validate beat detection method
		const allowedMethods = ["auto", "librosa", "madmom"];
		const beatDetection = String(settingsObj.beatDetection).toLowerCase();
		if (!allowedMethods.includes(beatDetection)) {
			return null;
		}

		return {
			introLength,
			outroLength,
			preserveVocals,
			beatDetection,
			algorithm:
				typeof settingsObj.algorithm === "string"
					? settingsObj.algorithm
					: undefined,
			tempo:
				typeof settingsObj.tempo === "number" ? settingsObj.tempo : undefined,
			key: typeof settingsObj.key === "string" ? settingsObj.key : undefined,
		};
	} catch (error) {
		console.error("Settings validation error:", error);
		return null;
	}
}

/**
 * Create safe Python execution options with parameter validation
 * @param scriptName - Name of Python script to run
 * @param args - Array of arguments (will be validated)
 * @returns Safe execution options
 */
function createSafePythonOptions(scriptName: string, args: string[]) {
	// Validate script name
	const allowedScripts = ["audioProcessor.py", "utils.py"];
	if (!allowedScripts.includes(scriptName)) {
		throw new Error("Invalid script name");
	}

	// Validate and sanitize arguments
	const sanitizedArgs = args.map((arg) => {
		if (typeof arg !== "string") {
			throw new Error("All arguments must be strings");
		}
		// Remove any potentially dangerous characters
		return arg.replace(/[;&|`$(){}[\]<>]/g, "");
	});

	// Use environment variable for Python path if available
	const pythonPath =
		process.env.PYTHON_PATH ||
		(process.platform === "win32" ? "python" : "python3");

	return {
		mode: "text" as const,
		pythonPath,
		pythonOptions: ["-u"],
		scriptPath: path.join(process.cwd(), "server"),
		args: sanitizedArgs,
	};
}

export async function registerRoutes(app: Express): Promise<Server> {
	const httpServer = createServer(app);

	// Log system startup
	await auditLogger.logSystemEvent(AuditEventType.SERVICE_START, {
		timestamp: new Date().toISOString(),
		nodeEnv: process.env.NODE_ENV || "development",
		auditConfig: {
			enabled: AUDIT_CONFIG.ENABLED,
			logToFile: AUDIT_CONFIG.LOG_TO_FILE,
			logToConsole: AUDIT_CONFIG.LOG_TO_CONSOLE,
		},
	});

	/**
	 * Route Handlers Documentation
	 *
	 * POST /api/tracks/upload
	 * - Handles audio file upload
	 * - Creates track entry in database
	 * - Analyzes audio for basic info (format, tempo, key)
	 *
	 * GET /api/tracks/:id
	 * - Retrieves specific track information
	 *
	 * GET /api/tracks
	 * - Lists all tracks for demo user
	 *
	 * DELETE /api/tracks
	 * - Clears all tracks and associated files
	 *
	 * POST /api/tracks/:id/process
	 * - Processes track to create extended version
	 * - Handles versioning and status updates
	 *
	 * GET /api/tracks/:id/status
	 * - Returns current processing status
	 *
	 * GET /api/audio/:id/:type
	 * - Streams audio files (original or extended)
	 *
	 * GET /api/tracks/:id/download
	 * - Handles download of processed tracks
	 */

	// Upload audio file
	app.post(
		"/api/tracks/upload",
		upload.single("audio"),
		async (req: Request & { file?: Express.Multer.File }, res: Response) => {
			try {
				if (!req.file) {
					await auditLogger.logSecurityViolation(
						req,
						AuditEventType.FILE_UPLOAD_REJECTED,
						{
							reason: "No file uploaded in request",
							endpoint: "/api/tracks/upload",
						}
					);
					return res.status(400).json({ message: "No file uploaded" });
				}

				// Security: Validate uploaded file path
				const safeUploadPath = validateAndSanitizePath(req.file.path);
				if (!safeUploadPath) {
					// Log security violation for path traversal attempt
					await auditLogger.logSecurityViolation(
						req,
						AuditEventType.PATH_TRAVERSAL_ATTEMPT,
						{
							filename: req.file.originalname,
							attemptedPath: req.file.path,
							reason: "File path validation failed",
						}
					);

					// Clean up unsafe file
					if (fs.existsSync(req.file.path)) {
						fs.unlinkSync(req.file.path);
					}
					return res.status(400).json({
						message: "Invalid upload path - file rejected",
					});
				}

				const track = await storage.createAudioTrack({
					originalFilename: req.file.originalname,
					originalPath: safeUploadPath,
					userId: demoUser.id, // Using demo user for now
				});

				// Get basic audio info using safe Python options
				const audioAnalysisArgs = [safeUploadPath];
				const options = createSafePythonOptions("utils.py", audioAnalysisArgs);

				PythonShell.run("utils.py", options)
					.then(async (results) => {
						if (results && results.length > 0) {
							try {
								const audioInfo = JSON.parse(results[0]);
								await storage.updateAudioTrack(track.id, {
									format: audioInfo.format,
									bitrate: audioInfo.bitrate || null,
									duration: audioInfo.duration || null,
									bpm: audioInfo.bpm || null,
									key: audioInfo.key || null,
								});
							} catch (e) {
								console.error("Error parsing audio info:", e);
							}
						}
					})
					.catch((err) => {
						console.error("Error analyzing audio:", err);
					});

				// Log the successful file upload event
				await auditLogger.logFileUpload(
					req,
					req.file.originalname,
					req.file.size,
					"SUCCESS"
				);

				return res.status(201).json(track);
			} catch (error) {
				// Log upload failure event
				await auditLogger.logFileUpload(
					req,
					req.file?.originalname || "unknown",
					req.file?.size || 0,
					"FAILURE",
					error instanceof Error ? error.message : "Unknown error"
				);

				// Log detailed error server-side only
				console.error("Upload error:", error);
				return res.status(500).json({
					message: "Upload failed - please check your file and try again",
				});
			}
		}
	);

	// Get a specific track
	app.get("/api/tracks/:id", async (req: Request, res: Response) => {
		try {
			const id = parseInt(req.params.id, 10);
			if (isNaN(id)) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid track ID parameter",
						providedId: req.params.id,
						endpoint: "/api/tracks/:id",
					}
				);
				return res.status(400).json({ message: "Invalid track ID" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				await auditLogger.log(
					AuditEventType.UNAUTHORIZED_ACCESS,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Track not found",
						trackId: id,
						endpoint: "/api/tracks/:id",
					},
					req
				);
				return res.status(404).json({ message: "Track not found" });
			}

			// Log successful track access
			await auditLogger.log(
				AuditEventType.FILE_ACCESS,
				AuditSeverity.LOW,
				"SUCCESS",
				{
					trackId: id,
					filename: track.originalFilename,
				},
				req
			);

			return res.json(track);
		} catch (error) {
			// Log system error
			await auditLogger.log(
				AuditEventType.SYSTEM_ERROR,
				AuditSeverity.MEDIUM,
				"FAILURE",
				{
					error: error instanceof Error ? error.message : "Unknown error",
					endpoint: "/api/tracks/:id",
				},
				req
			);

			// Log detailed error server-side only
			console.error("Get track error:", error);
			return res
				.status(500)
				.json({ message: "Unable to retrieve track information" });
		}
	});

	// Get all tracks for the demo user
	app.get("/api/tracks", async (req: Request, res: Response) => {
		try {
			const tracks = await storage.getAudioTracksByUserId(demoUser.id);
			return res.json(tracks);
		} catch (error) {
			// Log detailed error server-side only
			console.error("Get tracks error:", error);
			return res.status(500).json({ message: "Unable to retrieve tracks" });
		}
	});

	// Clear all tracks
	app.delete("/api/tracks", async (req: Request, res: Response) => {
		try {
			const tracks = await storage.getAudioTracksByUserId(demoUser.id);

			// Log the delete operation before starting
			await auditLogger.log(
				AuditEventType.FILE_DELETE,
				AuditSeverity.MEDIUM,
				"SUCCESS",
				{
					action: "clear_all_tracks",
					trackCount: tracks.length,
					userId: demoUser.id,
				},
				req
			);

			// Delete files
			for (const track of tracks) {
				// Security: Use safe deletion for original file
				safeDeleteFile(track.originalPath);

				// Security: Use safe deletion for extended files
				if (Array.isArray(track.extendedPaths)) {
					for (const filePath of track.extendedPaths) {
						safeDeleteFile(filePath as string);
					}
				}
			}

			// Delete from database
			await storage.deleteAllUserTracks(demoUser.id);

			return res.json({ message: "All tracks cleared" });
		} catch (error) {
			// Log detailed error server-side only
			console.error("Clear tracks error:", error);
			return res.status(500).json({ message: "Unable to clear tracks" });
		}
	});

	// Process a track to create extended version
	app.post("/api/tracks/:id/process", async (req: Request, res: Response) => {
		try {
			const id = parseInt(req.params.id, 10);
			if (isNaN(id)) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid track ID parameter",
						providedId: req.params.id,
						endpoint: "/api/tracks/:id/process",
					}
				);
				return res.status(400).json({ message: "Invalid track ID" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				await auditLogger.log(
					AuditEventType.UNAUTHORIZED_ACCESS,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Track not found for processing",
						trackId: id,
						endpoint: "/api/tracks/:id/process",
					},
					req
				);
				return res.status(404).json({ message: "Track not found" });
			}

			// Check version limit
			if (track.versionCount > 3) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Version limit exceeded",
						trackId: id,
						currentVersionCount: track.versionCount,
						maxAllowed: 3,
					}
				);
				return res.status(400).json({
					message: "Maximum version limit (3) reached",
				});
			}

			// Validate and sanitize settings from request
			const rawSettings = processingSettingsSchema.parse(req.body);
			const validatedSettings = validateProcessingSettings(rawSettings);

			if (!validatedSettings) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid processing settings",
						trackId: id,
						providedSettings: rawSettings,
					}
				);
				return res.status(400).json({
					message: "Invalid processing settings provided",
				});
			}

			// Log the start of audio processing
			await auditLogger.logProcessingEvent(
				id,
				AuditEventType.AUDIO_PROCESSING_START,
				"SUCCESS",
				{
					settings: validatedSettings,
					filename: track.originalFilename,
				}
			);

			// Update track status and settings
			await storage.updateAudioTrack(id, {
				status:
					Array.isArray(track.extendedPaths) && track.extendedPaths.length > 0
						? "regenerate"
						: "processing",
				settings: validatedSettings,
			});

			// Generate a filename for the extended version
			const outputBase = path.basename(
				track.originalFilename,
				path.extname(track.originalFilename)
			);
			const fileExt = path.extname(track.originalFilename);
			const version = Array.isArray(track.extendedPaths)
				? track.extendedPaths.length
				: 0;

			// Security: Create safe output path using sanitized filename
			const safeFilename = `${outputBase}_extended_v${version + 1}${fileExt}`;
			const outputPath = createSafeOutputPath(resultDir, safeFilename);

			if (!outputPath) {
				return res.status(400).json({
					message: "Invalid output path - unable to create safe file path",
				});
			}

			// Execute the Python script for audio processing with validated parameters
			const processingArgs = [
				track.originalPath,
				outputPath,
				validatedSettings.introLength.toString(),
				validatedSettings.outroLength.toString(),
				validatedSettings.preserveVocals.toString(),
				validatedSettings.beatDetection,
			];
			const options = createSafePythonOptions(
				"audioProcessor.py",
				processingArgs
			);

			// Send initial response
			res.status(202).json({
				message: "Processing started",
				trackId: id,
				status: "processing",
			});

			// Start processing in background
			PythonShell.run("audioProcessor.py", options)
				.then(async (results) => {
					console.log("Processing complete:", results);

					// Get audio info of the processed file using safe Python options
					const audioInfoArgs = [outputPath];
					const audioInfoOptions = createSafePythonOptions(
						"utils.py",
						audioInfoArgs
					);

					return PythonShell.run("utils.py", audioInfoOptions).then(
						async (infoResults) => {
							let extendedDuration = null;

							if (infoResults && infoResults.length > 0) {
								try {
									const audioInfo = JSON.parse(infoResults[0]);
									console.log("Extended audio info:", audioInfo);
									extendedDuration = audioInfo.duration || null;
								} catch (e) {
									console.error("Error parsing extended audio info:", e);
								}
							}

							// Update track with completed status and add new version
							const track = await storage.getAudioTrack(id);
							if (!track) {
								throw new Error("Track not found after processing");
							}

							const currentPaths = Array.isArray(track.extendedPaths)
								? track.extendedPaths
								: [];
							const currentDurations = Array.isArray(track.extendedDurations)
								? track.extendedDurations
								: [];
							let extendedPaths = [...currentPaths, outputPath];
							console.log("extendedPaths:", extendedPaths);

							// Log the successful audio processing event
							await auditLogger.logProcessingEvent(
								id,
								AuditEventType.AUDIO_PROCESSING_SUCCESS,
								"SUCCESS",
								{
									outputPath,
									extendedDuration,
								}
							);

							return storage.updateAudioTrack(id, {
								status: "completed",
								extendedPaths: extendedPaths,
								extendedDurations: [...currentDurations, extendedDuration],
								versionCount: (track.versionCount || 1) + 1,
							});
						}
					);
				})
				.catch(async (error) => {
					// Log detailed error server-side only
					console.error("Processing error:", error);
					await storage.updateAudioTrack(id, {
						status: "error",
					});

					// Log the failed audio processing event
					await auditLogger.logProcessingEvent(
						id,
						AuditEventType.AUDIO_PROCESSING_FAILURE,
						"FAILURE",
						{
							error: error.message,
						}
					);
				});
		} catch (error) {
			// Log detailed error server-side only
			console.error("Process track error:", error);
			return res
				.status(500)
				.json({ message: "Audio processing failed - please try again" });
		}
	});

	// Get processing status
	app.get("/api/tracks/:id/status", async (req: Request, res: Response) => {
		try {
			const id = parseInt(req.params.id, 10);
			if (isNaN(id)) {
				return res.status(400).json({ message: "Invalid track ID" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			return res.json({ status: track.status });
		} catch (error) {
			// Log detailed error server-side only
			console.error("Get status error:", error);
			return res
				.status(500)
				.json({ message: "Unable to retrieve processing status" });
		}
	});

	// Serve audio files
	app.get("/api/audio/:id/:type", async (req: Request, res: Response) => {
		try {
			const id = parseInt(req.params.id, 10);
			if (isNaN(id)) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid track ID parameter",
						providedId: req.params.id,
						endpoint: "/api/audio/:id/:type",
					}
				);
				return res.status(400).json({ message: "Invalid track ID" });
			}

			const type = req.params.type;
			if (type !== "original" && type !== "extended") {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid audio type parameter",
						providedType: type,
						endpoint: "/api/audio/:id/:type",
					}
				);
				return res.status(400).json({ message: "Invalid audio type" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Track not found",
						trackId: id,
						requestedType: type,
						endpoint: "/api/audio/:id/:type",
					},
					req
				);
				return res.status(404).json({ message: "Track not found" });
			}

			let filePath = track.originalPath;
			if (type === "extended") {
				const version = parseInt(req.query.version as string) || 0;
				const extendedPaths = Array.isArray(track.extendedPaths)
					? track.extendedPaths
					: [];
				filePath = extendedPaths[version];
			}

			if (!filePath) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: `${type} audio file path not found`,
						trackId: id,
						requestedType: type,
						version: req.query.version,
					},
					req
				);
				return res
					.status(404)
					.json({ message: `${type} audio file not found` });
			}

			// Security: Validate and sanitize the file path
			const safePath = validateAndSanitizePath(filePath);
			if (!safePath) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.PATH_TRAVERSAL_ATTEMPT,
					{
						reason: "File path validation failed",
						trackId: id,
						attemptedPath: filePath,
						requestedType: type,
					}
				);
				return res.status(400).json({
					message: "Invalid file path - path validation failed",
				});
			}

			if (!fs.existsSync(safePath)) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Audio file not found on disk",
						trackId: id,
						filePath: path.basename(safePath),
						requestedType: type,
					},
					req
				);
				return res
					.status(404)
					.json({ message: "Audio file not found on disk" });
			}

			// Log successful file access
			await auditLogger.logFileAccess(req, filePath, "SUCCESS");

			const stat = fs.statSync(safePath);
			const fileSize = stat.size;
			const range = req.headers.range;

			if (range) {
				const parts = range.replace(/bytes=/, "").split("-");
				const start = parseInt(parts[0], 10);
				const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
				const chunksize = end - start + 1;
				const file = fs.createReadStream(safePath, { start, end });
				const head = {
					"Content-Range": `bytes ${start}-${end}/${fileSize}`,
					"Accept-Ranges": "bytes",
					"Content-Length": chunksize,
					"Content-Type": "audio/mpeg",
				};
				res.writeHead(206, head);
				file.pipe(res);
			} else {
				const head = {
					"Content-Length": fileSize,
					"Content-Type": "audio/mpeg",
				};
				res.writeHead(200, head);
				fs.createReadStream(safePath).pipe(res);
			}

			// Log the successful file access event
			await auditLogger.logFileAccess(req, filePath, "SUCCESS");
		} catch (error) {
			// Log detailed error server-side only
			console.error("Stream audio error:", error);
			return res.status(500).json({
				message: "Unable to stream audio file",
			});
		}
	});

	// Download extended audio
	app.get("/api/tracks/:id/download", async (req: Request, res: Response) => {
		try {
			const id = parseInt(req.params.id, 10);
			if (isNaN(id)) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.SUSPICIOUS_REQUEST,
					{
						reason: "Invalid track ID parameter",
						providedId: req.params.id,
						endpoint: "/api/tracks/:id/download",
					}
				);
				return res.status(400).json({ message: "Invalid track ID" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Track not found for download",
						trackId: id,
						endpoint: "/api/tracks/:id/download",
					},
					req
				);
				return res.status(404).json({ message: "Track not found" });
			}

			const version = parseInt(req.query.version as string) || 0;
			const extendedPaths = Array.isArray(track.extendedPaths)
				? track.extendedPaths
				: [];

			if (version >= extendedPaths.length || !extendedPaths[version]) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Extended version not found",
						trackId: id,
						requestedVersion: version,
						availableVersions: extendedPaths.length,
					},
					req
				);
				return res.status(404).json({ message: "Extended version not found" });
			}

			const filePath = extendedPaths[version];

			// Security: Validate and sanitize the file path
			const safePath = validateAndSanitizePath(filePath);
			if (!safePath) {
				await auditLogger.logSecurityViolation(
					req,
					AuditEventType.PATH_TRAVERSAL_ATTEMPT,
					{
						reason: "File path validation failed for download",
						trackId: id,
						attemptedPath: filePath,
						version: version,
					}
				);
				return res.status(400).json({
					message: "Invalid file path - path validation failed",
				});
			}

			if (!fs.existsSync(safePath)) {
				await auditLogger.log(
					AuditEventType.FILE_ACCESS_DENIED,
					AuditSeverity.MEDIUM,
					"FAILURE",
					{
						reason: "Extended audio file not found on disk",
						trackId: id,
						filePath: path.basename(safePath),
						version: version,
					},
					req
				);
				return res
					.status(404)
					.json({ message: "Extended audio file not found on disk" });
			}

			// Extract original filename without extension
			const originalNameWithoutExt = path.basename(
				track.originalFilename,
				path.extname(track.originalFilename)
			);

			// Create download filename with version number
			const downloadFilename = `${originalNameWithoutExt}_extended_v${
				version + 1
			}${path.extname(track.originalFilename)}`;

			res.download(safePath, downloadFilename);

			// Log the successful file download event
			await auditLogger.logFileAccess(req, filePath, "SUCCESS");
		} catch (error) {
			// Log system error for download failure
			await auditLogger.log(
				AuditEventType.SYSTEM_ERROR,
				AuditSeverity.MEDIUM,
				"FAILURE",
				{
					error: error instanceof Error ? error.message : "Unknown error",
					endpoint: "/api/tracks/:id/download",
					trackId: req.params.id,
				},
				req
			);

			// Log detailed error server-side only
			console.error("Download error:", error);
			return res.status(500).json({
				message: "Unable to download file",
			});
		}
	});

	return httpServer;
}
