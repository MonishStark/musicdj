/**
 * Simplified Job Queue for Development (Fallback Mode)
 *
 * This is a minimal implementation that processes jobs directly when Redis is not available.
 * For production use, Redis should be properly configured for the full job queue system.
 *
 * @format
 */

import { storage } from "./storage";
import { ProcessingSettings } from "@shared/schema";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

// Security function to sanitize data for logging
function sanitizeForLog(data: any): string {
	if (typeof data === "string") {
		// Remove format specifiers and control characters that could manipulate log output
		return data
			.replace(/%[sdifj%]/g, "") // Remove format specifiers
			.replace(/[\x00-\x1f\x7f-\x9f]/g, "") // Remove control characters
			.slice(0, 1000); // Limit length
	}
	return String(data).slice(0, 1000);
}

// Job priority levels
export enum JobPriority {
	LOW = 1,
	NORMAL = 2,
	HIGH = 3,
	CRITICAL = 4,
}

// Job status types
export enum JobStatus {
	WAITING = "waiting",
	ACTIVE = "active",
	COMPLETED = "completed",
	FAILED = "failed",
	DELAYED = "delayed",
	PAUSED = "paused",
}

// Job data interfaces
export interface AudioProcessingJobData {
	jobId: string;
	trackId: number;
	originalPath: string;
	outputPath: string;
	settings: ProcessingSettings;
	userId: number;
	priority: JobPriority;
	retryAttempt?: number;
	useOptimization?: boolean;
}

export interface JobProgress {
	percentage: number;
	stage: string;
	message: string;
	currentStep: number;
	totalSteps: number;
	estimatedTimeRemaining?: number;
	memoryUsage?: number;
}

/**
 * Simplified Job Queue Manager (Development/Fallback Mode)
 *
 * Processes jobs directly without Redis queue for development purposes.
 */
class SimpleJobQueueManager {
	private activeJobs: Map<string, any> = new Map();
	private socketIo?: any;

	constructor() {
		console.log(
			"📝 Job Queue running in direct processing mode (Redis not available)"
		);
	}

	setSocketIo(io: any) {
		this.socketIo = io;
	}

	/**
	 * Add audio processing job (direct processing)
	 */
	async addAudioProcessingJob(data: AudioProcessingJobData): Promise<string> {
		console.log(
			`🎵 Starting direct audio processing for track ${data.trackId}`
		);

		// Store job info
		this.activeJobs.set(data.jobId, {
			...data,
			status: JobStatus.ACTIVE,
			progress: {
				percentage: 0,
				stage: "starting",
				message: "Initializing...",
				currentStep: 1,
				totalSteps: 6,
			},
		});

		// Process immediately in background
		this.processAudioJob(data).catch((error) => {
			console.error("❌ Direct processing failed:", error);
			this.activeJobs.set(data.jobId, {
				...data,
				status: JobStatus.FAILED,
				error: error.message,
			});
		});

		return data.jobId;
	}

	/**
	 * Get job status
	 */
	async getJobStatus(jobId: string) {
		const job = this.activeJobs.get(jobId);
		if (!job) {
			return { status: "not_found" };
		}

		return {
			jobId,
			status: job.status,
			progress: job.progress,
			error: job.error,
		};
	}

	/**
	 * Cancel job (mark as cancelled)
	 */
	async cancelJob(jobId: string): Promise<boolean> {
		const job = this.activeJobs.get(jobId);
		if (job && job.status === JobStatus.ACTIVE) {
			job.status = JobStatus.FAILED;
			job.error = "Cancelled by user";
			console.log(`🚫 Job ${jobId} marked as cancelled`);
			return true;
		}
		return false;
	}

	/**
	 * Process audio job directly
	 */
	private async processAudioJob(data: AudioProcessingJobData): Promise<void> {
		const updateProgress = (progress: Partial<JobProgress>) => {
			const job = this.activeJobs.get(data.jobId);
			if (job) {
				job.progress = { ...job.progress, ...progress };

				// Emit progress via WebSocket if available
				if (this.socketIo) {
					this.socketIo.to(`user-${data.userId}`).emit("job-update", {
						jobId: data.jobId,
						trackId: data.trackId,
						userId: data.userId,
						status: job.status,
						progress: job.progress,
						timestamp: new Date().toISOString(),
					});
				}
			}
		};

		try {
			// Update database status
			await storage.updateAudioTrack(data.trackId, { status: "processing" });

			// Step 1: Setup
			updateProgress({
				percentage: 10,
				stage: "setup",
				message: "Setting up processing environment...",
				currentStep: 1,
			});

			// Step 2: File validation
			updateProgress({
				percentage: 20,
				stage: "validation",
				message: "Validating audio file...",
				currentStep: 2,
			});

			if (!fs.existsSync(data.originalPath)) {
				throw new Error("Source audio file not found");
			}

			// Step 3: Processing
			updateProgress({
				percentage: 30,
				stage: "processing",
				message: "Processing audio...",
				currentStep: 3,
			});

			// Determine which Python script to use
			const scriptName = data.useOptimization
				? "audioProcessor_optimized.py"
				: "audioProcessor.py";
			const scriptPath = path.join(__dirname, scriptName);

			// Check if optimized script exists, fallback to regular if not
			const finalScriptPath = fs.existsSync(scriptPath)
				? scriptPath
				: path.join(__dirname, "audioProcessor.py");

			// Run Python processing
			const result = await this.runPythonScript(
				finalScriptPath,
				data,
				updateProgress
			);

			// Step 4: Validation
			updateProgress({
				percentage: 80,
				stage: "validation",
				message: "Validating output...",
				currentStep: 5,
			});

			if (!fs.existsSync(data.outputPath)) {
				throw new Error("Processing completed but output file not found");
			}

			// Step 5: Finalization
			updateProgress({
				percentage: 90,
				stage: "finalizing",
				message: "Finalizing...",
				currentStep: 6,
			});

			// Update database
			await storage.updateAudioTrack(data.trackId, {
				status: "completed",
				processedFilePath: data.outputPath,
			});

			// Complete
			updateProgress({
				percentage: 100,
				stage: "completed",
				message: "Processing completed successfully!",
				currentStep: 6,
			});

			const job = this.activeJobs.get(data.jobId);
			if (job) {
				job.status = JobStatus.COMPLETED;
			}

			console.log(`✅ Direct processing completed for track ${data.trackId}`);
		} catch (error) {
			console.error(
				// nosemgrep: javascript.lang.security.audit.unsafe-formatstring.unsafe-formatstring
				`❌ Direct processing failed for track ${sanitizeForLog(
					data.trackId
				)}:`,
				error
			);

			// Update database
			await storage.updateAudioTrack(data.trackId, { status: "error" });

			// Update job status
			const job = this.activeJobs.get(data.jobId);
			if (job) {
				job.status = JobStatus.FAILED;
				job.error = error.message;
			}

			throw error;
		}
	}

	/**
	 * Run Python script for audio processing
	 */
	private async runPythonScript(
		scriptPath: string,
		data: AudioProcessingJobData,
		updateProgress: (progress: Partial<JobProgress>) => void
	): Promise<any> {
		return new Promise((resolve, reject) => {
			const args = [
				scriptPath,
				data.originalPath,
				data.outputPath,
				data.settings.introLength?.toString() || "16",
				data.settings.outroLength?.toString() || "16",
				data.settings.preserveVocals ? "true" : "false",
				data.settings.beatDetection || "auto",
			];

			console.log("🐍 Running Python script:", args.join(" "));

			const pythonProcess = spawn("python", args, {
				cwd: __dirname,
				stdio: ["ignore", "pipe", "pipe"],
			});

			let stdout = "";
			let stderr = "";
			let lastProgress = 30;

			pythonProcess.stdout?.on("data", (data) => {
				stdout += data.toString();

				// Simulate progress updates
				lastProgress = Math.min(75, lastProgress + 5);
				updateProgress({
					percentage: lastProgress,
					stage: "processing",
					message: "Audio processing in progress...",
					currentStep: 4,
				});
			});

			pythonProcess.stderr?.on("data", (data) => {
				stderr += data.toString();
			});

			pythonProcess.on("close", (code) => {
				if (code === 0) {
					console.log("✅ Python script completed successfully");
					resolve({ stdout, stderr });
				} else {
					console.error("❌ Python script failed with code:", code);
					console.error("Error output:", stderr);
					reject(
						new Error(`Python script failed with exit code ${code}: ${stderr}`)
					);
				}
			});

			pythonProcess.on("error", (error) => {
				console.error("❌ Failed to start Python script:", error);
				reject(new Error(`Failed to start Python script: ${error.message}`));
			});
		});
	}

	/**
	 * Get queue statistics (simplified)
	 */
	async getQueueStats() {
		const active = Array.from(this.activeJobs.values()).filter(
			(job) => job.status === JobStatus.ACTIVE
		).length;
		const completed = Array.from(this.activeJobs.values()).filter(
			(job) => job.status === JobStatus.COMPLETED
		).length;
		const failed = Array.from(this.activeJobs.values()).filter(
			(job) => job.status === JobStatus.FAILED
		).length;

		return {
			timestamp: new Date().toISOString(),
			queues: {
				audioProcessing: {
					waiting: 0,
					active,
					completed,
					failed,
					total: this.activeJobs.size,
				},
			},
			summary: {
				totalJobs: this.activeJobs.size,
				activeJobs: active,
				waitingJobs: 0,
				failedJobs: failed,
			},
		};
	}

	/**
	 * Health check
	 */
	async getHealth() {
		const stats = await this.getQueueStats();
		const failureRate =
			stats.summary.totalJobs > 0
				? ((stats.summary.failedJobs / stats.summary.totalJobs) * 100).toFixed(
						2
				  )
				: "0.00";

		return {
			status: "healthy",
			mode: "direct-processing",
			timestamp: new Date().toISOString(),
			metrics: {
				totalJobs: stats.summary.totalJobs,
				activeJobs: stats.summary.activeJobs,
				failedJobs: stats.summary.failedJobs,
				failureRate: `${failureRate}%`,
			},
			thresholds: {
				maxFailureRate: "10%",
				maxActiveJobs: 20,
			},
		};
	}

	/**
	 * Cleanup old jobs from memory
	 */
	cleanupOldJobs() {
		const oneHourAgo = Date.now() - 60 * 60 * 1000;
		let cleaned = 0;

		for (const [jobId, job] of this.activeJobs.entries()) {
			if (
				job.createdAt &&
				job.createdAt < oneHourAgo &&
				(job.status === JobStatus.COMPLETED || job.status === JobStatus.FAILED)
			) {
				this.activeJobs.delete(jobId);
				cleaned++;
			}
		}

		if (cleaned > 0) {
			console.log(`🧹 Cleaned up ${cleaned} old jobs from memory`);
		}
	}

	/**
	 * Graceful shutdown
	 */
	async shutdown() {
		console.log("🔄 Shutting down job queue manager...");

		// Cancel active jobs
		for (const [jobId, job] of this.activeJobs.entries()) {
			if (job.status === JobStatus.ACTIVE) {
				job.status = JobStatus.FAILED;
				job.error = "Server shutdown";
			}
		}

		console.log("✅ Job queue manager shutdown completed");
	}

	// Placeholder methods for compatibility
	async addAudioAnalysisJob(data: any): Promise<string> {
		console.log("📊 Audio analysis not available in fallback mode");
		return data.jobId;
	}

	async addFileCleanupJob(data: any): Promise<string> {
		console.log("🧹 File cleanup will be handled manually in fallback mode");
		return data.jobId;
	}

	async addNotificationJob(
		userId: number,
		trackId: number,
		type: string,
		message: string
	): Promise<string> {
		console.log(`📢 Notification (${type}): ${message}`);
		return "notification-" + Date.now();
	}

	async pauseAllQueues(): Promise<void> {
		console.log("⏸️ Queue pause not available in fallback mode");
	}

	async resumeAllQueues(): Promise<void> {
		console.log("▶️ Queue resume not available in fallback mode");
	}
}

// Create and export the job queue manager instance
export const jobQueueManager = new SimpleJobQueueManager();

// Cleanup interval
setInterval(() => {
	jobQueueManager.cleanupOldJobs();
}, 30 * 60 * 1000); // Every 30 minutes

export default jobQueueManager;
