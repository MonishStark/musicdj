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
import streamingRoutes from "./streaming-routes.js";
import {
	SecurePathValidator,
	InputSanitizer,
	createSecurityMiddleware,
} from "./security-utils.js";

// Setup multer for file uploads with proper validation
const uploadsDir =
	process.env.UPLOADS_DIR || path.join(process.cwd(), "uploads");
const resultDir =
	process.env.RESULTS_DIR || path.join(process.cwd(), "results");

// Initialize security components
const allowedDirectories = [uploadsDir, resultDir];
const secureValidator = new SecurePathValidator(allowedDirectories);
const securityMiddleware = createSecurityMiddleware(secureValidator);

// Security: Validate and canonicalize directory paths to prevent malicious paths
let normalizedUploadsDir: string;
let normalizedResultDir: string;

try {
	normalizedUploadsDir = validateAndCanonicalizeDirectory(uploadsDir);
	normalizedResultDir = validateAndCanonicalizeDirectory(resultDir);
} catch (error) {
	console.error("Directory validation failed:", error);
	throw error;
}

// Ensure directories exist and are properly secured
if (!fs.existsSync(normalizedUploadsDir)) {
	fs.mkdirSync(normalizedUploadsDir, { recursive: true });
}
if (!fs.existsSync(normalizedResultDir)) {
	fs.mkdirSync(normalizedResultDir, { recursive: true });
}

// Security utility function to validate file paths and prevent path traversal
function validateFilePath(filePath: string, allowedDirectory: string): boolean {
	try {
		// Step 1: Canonicalize both paths to their absolute forms
		// This resolves all relative components (., .., symlinks, etc.)
		const canonicalFilePath = path.resolve(filePath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal
		const canonicalBaseDirectory = path.resolve(allowedDirectory); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

		// Step 2: Ensure the canonicalized file path starts with the canonicalized base directory
		// This is the core defense against path traversal attacks
		const isWithinBaseDirectory =
			canonicalFilePath.startsWith(canonicalBaseDirectory + path.sep) ||
			canonicalFilePath === canonicalBaseDirectory;

		// Step 3: Additional security checks for common bypass attempts
		const containsDangerousPatterns =
			filePath.includes("..") || // Directory traversal
			filePath.includes("~") || // Home directory expansion
			filePath.includes("\0") || // Null byte injection
			filePath.includes("%00") || // URL encoded null byte
			filePath.includes("%2e%2e") || // URL encoded ..
			filePath.includes("%2f") || // URL encoded /
			filePath.includes("%5c"); // URL encoded \

		// Step 4: Verify the path doesn't contain any dangerous characters
		const hasInvalidChars = /[<>"|*?]/.test(filePath);

		return (
			isWithinBaseDirectory && !containsDangerousPatterns && !hasInvalidChars
		);
	} catch (error) {
		// If path resolution fails for any reason, deny access
		console.error("Path validation error:", error);
		return false;
	}
}

// Enhanced security function to validate and canonicalize directory paths
function validateAndCanonicalizeDirectory(dirPath: string): string {
	try {
		// Canonicalize the directory path
		const canonicalPath = path.resolve(dirPath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

		// Additional validation for directory paths
		if (canonicalPath.includes("..") || canonicalPath.includes("~")) {
			throw new Error(`Invalid directory path: ${dirPath}`);
		}

		return canonicalPath;
	} catch (error) {
		throw new Error(`Failed to validate directory path: ${dirPath}`);
	}
}

// Security function to validate file operations with canonicalized paths
function secureFileOperation(
	filePath: string,
	baseDirectory: string,
	operation: string
): boolean {
	try {
		// Canonicalize both paths
		const canonicalFilePath = path.resolve(filePath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal
		const canonicalBaseDir = path.resolve(baseDirectory); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

		// Verify the file is within the base directory
		const isWithinBase =
			canonicalFilePath.startsWith(canonicalBaseDir + path.sep) ||
			canonicalFilePath === canonicalBaseDir;

		if (!isWithinBase) {
			console.warn(
				`Security violation: ${operation} attempted outside base directory`
			);
			console.warn(`File path: ${canonicalFilePath}`);
			console.warn(`Base directory: ${canonicalBaseDir}`);
			return false;
		}

		return true;
	} catch (error) {
		// nosemgrep: javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
		console.error(`Security check failed for ${operation}:`, error);
		return false;
	}
}

// Security function to sanitize filename
function sanitizeFilename(filename: string): string {
	// Remove dangerous characters and path traversal attempts
	return filename
		.replace(/[<>:"/\\|?*\0]/g, "") // Remove filesystem-dangerous characters
		.replace(/\.\./g, "") // Remove path traversal attempts
		.replace(/^\.+/, "") // Remove leading dots
		.slice(0, 255); // Limit filename length
}
const storage_config = multer.diskStorage({
	destination: function (req, file, cb) {
		cb(null, normalizedUploadsDir);
	},
	filename: function (req, file, cb) {
		const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
		const sanitizedOriginalName = sanitizeFilename(file.originalname);
		const extension = path.extname(sanitizedOriginalName);
		cb(null, uniqueSuffix + extension);
	},
});

const upload = multer({
	storage: storage_config,
	limits: {
		fileSize: 15 * 1024 * 1024,
	},
	fileFilter: (req, file, cb) => {
		const allowedMimeTypes = [
			"audio/mpeg",
			"audio/wav",
			"audio/flac",
			"audio/aiff",
			"audio/x-aiff",
		];
		if (allowedMimeTypes.includes(file.mimetype)) {
			cb(null, true);
		} else {
			cb(
				new Error(
					"Invalid file type. Only MP3, WAV, FLAC, and AIFF files are allowed."
				)
			);
		}
	},
});

export async function registerRoutes(app: Express): Promise<Server> {
	const httpServer = createServer(app);

	// Set up user for demo purposes
	let demoUser = await storage.getUserByUsername("demo");
	if (!demoUser) {
		demoUser = await storage.createUser({
			username: "demo",
			password: "password", // In a real app, this would be hashed
		});
	}

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
		async (req: Request, res: Response) => {
			try {
				if (!req.file) {
					return res.status(400).json({ message: "No file uploaded" });
				}

				// Security: Validate uploaded file path
				if (!validateFilePath(req.file.path, normalizedUploadsDir)) {
					// Clean up the invalid file
					if (fs.existsSync(req.file.path)) {
						// nosemgrep: javascript.express.express-fs-filename.express-fs-filename
						// nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
						fs.unlinkSync(req.file.path); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal
					}
					return res
						.status(403)
						.json({ message: "Access denied: Invalid file path" });
				}

				const track = await storage.createAudioTrack({
					originalFilename: req.file.originalname,
					originalPath: req.file.path,
					userId: demoUser.id, // Using demo user for now
				});

				// Get basic audio info using Python
				const options = {
					mode: "text" as const,
					pythonPath: process.platform === "win32" ? "python" : "python3",
					pythonOptions: ["-u"],
					scriptPath: path.join(process.cwd(), "server"),
					args: [req.file.path],
				};

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

				return res.status(201).json(track);
			} catch (error) {
				console.error("Upload error:", error);
				const errorMessage =
					error instanceof Error ? error.message : "Unknown error";
				return res
					.status(500)
					.json({ message: "Error uploading file", error: errorMessage });
			}
		}
	);

	// Get a specific track
	app.get("/api/tracks/:id", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate and sanitize ID parameter
			const id = InputSanitizer.sanitizeIntParam(
				req.params.id,
				1,
				Number.MAX_SAFE_INTEGER
			);
			if (id === null) {
				return res.status(400).json({
					message: "Invalid track ID: must be a positive integer",
				});
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			return res.json(track);
		} catch (error) {
			console.error("Get track error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res
				.status(500)
				.json({ message: "Error retrieving track", error: errorMessage });
		}
	});

	// Get all tracks for the demo user
	app.get("/api/tracks", async (req: Request, res: Response) => {
		try {
			const tracks = await storage.getAudioTracksByUserId(demoUser.id);
			return res.json(tracks);
		} catch (error) {
			console.error("Get tracks error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res
				.status(500)
				.json({ message: "Error retrieving tracks", error: errorMessage });
		}
	});

	// Clear all tracks
	app.delete("/api/tracks", async (req: Request, res: Response) => {
		try {
			const tracks = await storage.getAudioTracksByUserId(demoUser.id);

			// Delete files with enhanced security validation
			for (const track of tracks) {
				// Validate and delete original file
				if (
					track.originalPath &&
					secureFileOperation(
						track.originalPath,
						normalizedUploadsDir,
						"delete"
					)
				) {
					if (fs.existsSync(track.originalPath)) {
						// nosemgrep: javascript.express.express-fs-filename.express-fs-filename
						// nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
						fs.unlinkSync(track.originalPath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal
					}
				}

				// Validate and delete extended files
				if (track.extendedPaths && Array.isArray(track.extendedPaths)) {
					for (const filePath of track.extendedPaths) {
						if (secureFileOperation(filePath, normalizedResultDir, "delete")) {
							if (fs.existsSync(filePath)) {
								// nosemgrep: javascript.express.express-fs-filename.express-fs-filename
								// nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
								fs.unlinkSync(filePath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal
							}
						}
					}
				}
			}

			// Delete from database
			await storage.deleteAllUserTracks(demoUser.id);

			return res.json({ message: "All tracks cleared" });
		} catch (error) {
			console.error("Clear tracks error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res
				.status(500)
				.json({ message: "Error clearing tracks", error: errorMessage });
		}
	});

	// Process a track to create extended version
	app.post("/api/tracks/:id/process", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate and sanitize ID parameter
			const id = InputSanitizer.sanitizeIntParam(
				req.params.id,
				1,
				Number.MAX_SAFE_INTEGER
			);
			if (id === null) {
				return res.status(400).json({
					message: "Invalid track ID: must be a positive integer",
				});
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			// Check version limit

			if (track.versionCount > 3) {
				return res.status(400).json({
					message: "Maximum version limit (3) reached",
				});
			}

			// Validate settings from request
			const settings = processingSettingsSchema.parse(req.body);

			// Update track status and settings
			await storage.updateAudioTrack(id, {
				status: (track.extendedPaths as string[])?.length
					? "regenerate"
					: "processing",
				settings: settings,
			});

			// Generate a filename for the extended version with security validation
			const outputBase = path.basename(
				track.originalFilename,
				path.extname(track.originalFilename)
			);
			const fileExt = path.extname(track.originalFilename);
			const version = (track.extendedPaths as string[])?.length || 0;
			const sanitizedBaseName = sanitizeFilename(outputBase);
			const outputFilename = `${sanitizedBaseName}_extended_v${
				version + 1
			}${fileExt}`;
			const outputPath = path.join(normalizedResultDir, outputFilename); // nosemgrep: javascript.express.security.audit.express-path-join-resolve-traversal.express-path-join-resolve-traversal, javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

			// Security: Validate the generated output path is within the results directory
			if (!validateFilePath(outputPath, normalizedResultDir)) {
				return res.status(500).json({
					message: "Error: Generated output path is invalid",
				});
			}

			// Execute the Python script for audio processing
			const options = {
				mode: "text" as const,
				pythonPath: process.platform === "win32" ? "python" : "python3",
				pythonOptions: ["-u"],
				scriptPath: path.join(process.cwd(), "server"),
				args: [
					track.originalPath,
					outputPath,
					settings.introLength.toString(),
					settings.outroLength.toString(),
					settings.preserveVocals.toString(),
					settings.beatDetection,
				],
			};

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

					// Get audio info of the processed file
					const audioInfoOptions = {
						mode: "text" as const,
						pythonPath: process.platform === "win32" ? "python" : "python3",
						pythonOptions: ["-u"],
						scriptPath: path.join(process.cwd(), "server"),
						args: [outputPath],
					};

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
								throw new Error("Track not found during completion update");
							}
							const currentPaths = (track.extendedPaths as string[]) || [];
							const currentDurations =
								(track.extendedDurations as number[]) || [];
							const extendedPaths = [...currentPaths, outputPath];
							console.log("extendedPaths:", extendedPaths);

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
					console.error("Processing error:", error);
					await storage.updateAudioTrack(id, {
						status: "error",
					});
				});
		} catch (error) {
			console.error("Process track error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res
				.status(500)
				.json({ message: "Error processing track", error: errorMessage });
		}
	});

	// Get processing status
	app.get("/api/tracks/:id/status", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate and sanitize ID parameter
			const id = InputSanitizer.sanitizeIntParam(
				req.params.id,
				1,
				Number.MAX_SAFE_INTEGER
			);
			if (id === null) {
				return res.status(400).json({
					message: "Invalid track ID: must be a positive integer",
				});
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			return res.json({ status: track.status });
		} catch (error) {
			console.error("Get status error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res
				.status(500)
				.json({ message: "Error retrieving status", error: errorMessage });
		}
	});

	// Serve audio files
	app.get("/api/audio/:id/:type", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate and sanitize ID parameter
			const id = InputSanitizer.sanitizeIntParam(
				req.params.id,
				1,
				Number.MAX_SAFE_INTEGER
			);
			if (id === null) {
				return res.status(400).json({
					message: "Invalid track ID: must be a positive integer",
				});
			}

			// Enhanced security: Validate and sanitize type parameter
			const type = InputSanitizer.sanitizeStringParam(req.params.type, [
				"original",
				"extended",
			]);
			if (!type) {
				return res.status(400).json({ message: "Invalid audio type" });
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			let filePath = track.originalPath;
			if (type === "extended") {
				// Enhanced security: Validate version parameter
				const version =
					InputSanitizer.sanitizeIntParam(
						req.query.version as string,
						0,
						100
					) || 0;
				const extendedPaths = Array.isArray(track.extendedPaths)
					? track.extendedPaths
					: [];
				filePath = extendedPaths[version];
			}

			if (!filePath) {
				return res
					.status(404)
					.json({ message: `${type} audio file not found` });
			}

			// Enhanced security: Comprehensive file path validation
			const pathValidation = secureValidator.validatePath(filePath, "read");
			if (!pathValidation.isValid) {
				console.warn("Path validation failed:", pathValidation.errors);
				return res.status(403).json({
					message: "Access denied: Invalid file path",
					errors: pathValidation.errors,
				});
			}

			// Security: Validate file path contains only safe characters and extensions
			const allowedExtensions = [".mp3", ".wav", ".flac", ".aiff"];
			if (!secureValidator.validateFileExtension(filePath, allowedExtensions)) {
				return res.status(400).json({ message: "Invalid file type" });
			}

			// Security: Enhanced path validation with canonicalization
			const isUploadFile =
				validateFilePath(filePath, normalizedUploadsDir) &&
				secureFileOperation(filePath, normalizedUploadsDir, "read");
			const isResultFile =
				validateFilePath(filePath, normalizedResultDir) &&
				secureFileOperation(filePath, normalizedResultDir, "read");

			if (!isUploadFile && !isResultFile) {
				return res
					.status(403)
					.json({ message: "Access denied: Invalid file path" });
			}

			// nosemgrep: javascript.express.express-fs-filename.express-fs-filename
			if (!fs.existsSync(filePath)) {
				return res
					.status(404)
					.json({ message: "Audio file not found on disk" });
			}

			const stat = fs.statSync(filePath); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
			const fileSize = stat.size;
			const range = req.headers.range;

			if (range) {
				const parts = range.replace(/bytes=/, "").split("-");
				const start = parseInt(parts[0], 10);
				const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
				const chunksize = end - start + 1;
				const file = fs.createReadStream(filePath, { start, end }); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
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
				fs.createReadStream(filePath).pipe(res); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal, javascript.express.express-fs-filename.express-fs-filename
			}
		} catch (error) {
			console.error("Stream audio error:", error);
			return res.status(500).json({
				message: "Error streaming audio",
				error: error instanceof Error ? error.message : "Unknown error",
			});
		}
	});

	// Download extended audio
	app.get("/api/tracks/:id/download", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate and sanitize ID parameter
			const id = InputSanitizer.sanitizeIntParam(
				req.params.id,
				1,
				Number.MAX_SAFE_INTEGER
			);
			if (id === null) {
				return res.status(400).json({
					message: "Invalid track ID: must be a positive integer",
				});
			}

			const track = await storage.getAudioTrack(id);
			if (!track) {
				return res.status(404).json({ message: "Track not found" });
			}

			// Enhanced security: Validate version parameter
			const version =
				InputSanitizer.sanitizeIntParam(req.query.version as string, 0, 100) ||
				0;
			const extendedPaths = Array.isArray(track.extendedPaths)
				? track.extendedPaths
				: [];

			if (version >= extendedPaths.length || !extendedPaths[version]) {
				return res.status(404).json({ message: "Extended version not found" });
			}

			const filePath = extendedPaths[version];

			// Security: Validate file path contains only safe characters and extensions
			const allowedExtensions = [".mp3", ".wav", ".flac", ".aiff"];
			const fileExtension = path.extname(filePath).toLowerCase();
			if (!allowedExtensions.includes(fileExtension)) {
				return res.status(400).json({ message: "Invalid file type" });
			}

			// Security: Enhanced path validation with canonicalization
			if (
				!validateFilePath(filePath, normalizedResultDir) ||
				!secureFileOperation(filePath, normalizedResultDir, "download")
			) {
				return res
					.status(403)
					.json({ message: "Access denied: Invalid file path" });
			}

			// nosemgrep: javascript.express.express-fs-filename.express-fs-filename
			if (!fs.existsSync(filePath)) {
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

			res.download(filePath, downloadFilename); // nosemgrep: javascript.lang.security.audit.path-traversal.path-join-resolve-traversal
		} catch (error) {
			console.error("Download error:", error);
			return res.status(500).json({
				message: "Error downloading file",
				error: error instanceof Error ? error.message : "Unknown error",
			});
		}
	});

	// Add streaming upload routes for large file support
	app.use("/api/streaming", streamingRoutes);

	return httpServer;
}
