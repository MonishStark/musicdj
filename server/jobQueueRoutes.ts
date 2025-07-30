/**
 * Enhanced Routes with Background Job Queue Integration
 *
 * This module extends the existing routes system to integrate with the background
 * job queue for improved audio processing performance and user experience.
 *
 * Key enhancements:
 * - Asynchronous job processing with immediate response
 * - Real-time progress tracking via WebSocket
 * - Job management endpoints (status, cancel, retry)
 * - Queue monitoring and statistics
 * - Integration with memory-optimized Python processing
 *
 * @format
 */

import type { Express, Request, Response } from "express";
import { jobQueueManager, JobPriority } from "./jobQueue";
import { storage } from "./storage";
import { processingSettingsSchema } from "@shared/schema";
import { InputSanitizer } from "./security-utils.js";
import path from "path";
import fs from "fs";

/**
 * Enhanced route handlers with job queue integration
 */
export function setupJobQueueRoutes(app: Express) {
	/**
	 * Enhanced audio processing endpoint with job queue integration
	 *
	 * POST /api/tracks/:id/process-async
	 * Starts audio processing as a background job with immediate response
	 */
	app.post(
		"/api/tracks/:id/process-async",
		async (req: Request, res: Response) => {
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
				const priority = req.body.priority || JobPriority.NORMAL;
				const useOptimization = req.body.useOptimization !== false; // Default to true

				// Update track status and settings
				await storage.updateAudioTrack(id, {
					status: (track.extendedPaths as string[])?.length
						? "regenerate"
						: "processing",
					settings: settings,
				});

				// Generate output path
				const outputBase = path.basename(
					track.originalFilename,
					path.extname(track.originalFilename)
				);
				const fileExt = path.extname(track.originalFilename);
				const version = (track.extendedPaths as string[])?.length || 0;
				const sanitizedBaseName = outputBase.replace(/[<>:"/\\|?*\0]/g, "");
				const outputFilename = `${sanitizedBaseName}_extended_v${
					version + 1
				}${fileExt}`;
				const resultDir =
					process.env.RESULTS_DIR || path.join(process.cwd(), "results");
				const outputPath = path.join(resultDir, outputFilename); // nosemgrep: javascript.express.security.audit.express-path-join-resolve-traversal.express-path-join-resolve-traversal, javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

				// Add job to queue
				const jobId = await jobQueueManager.addAudioProcessingJob(
					id,
					track.originalPath,
					outputPath,
					settings,
					track.userId || 1, // Default to demo user
					priority,
					useOptimization
				);

				// Return immediate response with job information
				res.status(202).json({
					message: "Audio processing job queued successfully",
					jobId,
					trackId: id,
					status: "queued",
					priority,
					useOptimization,
					estimatedProcessingTime: "2-5 minutes", // Estimate based on typical file sizes
				});
			} catch (error) {
				console.error("Process track async error:", error);
				const errorMessage =
					error instanceof Error ? error.message : "Unknown error";
				return res.status(500).json({
					message: "Error queueing processing job",
					error: errorMessage,
				});
			}
		}
	);

	/**
	 * Get job status endpoint
	 *
	 * GET /api/jobs/:jobId/status
	 * Returns detailed status information for a specific job
	 */
	app.get("/api/jobs/:jobId/status", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate job ID format
			const jobId = InputSanitizer.validateJobId(req.params.jobId);
			if (!jobId) {
				return res.status(400).json({
					message: "Invalid job ID format",
				});
			}

			const jobStatus = await jobQueueManager.getJobStatus(jobId);

			if (!jobStatus) {
				return res.status(404).json({ message: "Job not found" });
			}

			res.json({
				jobId,
				status: jobStatus.state,
				progress: jobStatus.progress,
				createdAt: jobStatus.createdAt,
				processedOn: jobStatus.processedOn,
				finishedOn: jobStatus.finishedOn,
				failedReason: jobStatus.failedReason,
				attemptsMade: jobStatus.attemptsMade,
				data: jobStatus.data,
			});
		} catch (error) {
			console.error("Get job status error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res.status(500).json({
				message: "Error retrieving job status",
				error: errorMessage,
			});
		}
	});

	/**
	 * Cancel job endpoint
	 *
	 * DELETE /api/jobs/:jobId
	 * Cancels a pending or active job
	 */
	app.delete("/api/jobs/:jobId", async (req: Request, res: Response) => {
		try {
			// Enhanced security: Validate job ID format
			const jobId = InputSanitizer.validateJobId(req.params.jobId);
			if (!jobId) {
				return res.status(400).json({
					message: "Invalid job ID format",
				});
			}

			const cancelled = await jobQueueManager.cancelJob(jobId);

			if (!cancelled) {
				return res
					.status(404)
					.json({ message: "Job not found or already completed" });
			}

			res.json({
				message: "Job cancelled successfully",
				jobId,
				cancelled: true,
			});
		} catch (error) {
			console.error("Cancel job error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res.status(500).json({
				message: "Error cancelling job",
				error: errorMessage,
			});
		}
	});

	/**
	 * Queue statistics endpoint
	 *
	 * GET /api/admin/queue-stats
	 * Returns comprehensive queue statistics for monitoring
	 */
	app.get("/api/admin/queue-stats", async (req: Request, res: Response) => {
		try {
			// In production, add proper authentication and admin role checking here

			const stats = await jobQueueManager.getQueueStats();

			res.json({
				timestamp: new Date().toISOString(),
				queues: stats,
				summary: {
					totalJobs: Object.values(stats).reduce(
						(sum: number, queue: any) => sum + queue.total,
						0
					),
					activeJobs: Object.values(stats).reduce(
						(sum: number, queue: any) => sum + queue.active,
						0
					),
					waitingJobs: Object.values(stats).reduce(
						(sum: number, queue: any) => sum + queue.waiting,
						0
					),
					failedJobs: Object.values(stats).reduce(
						(sum: number, queue: any) => sum + queue.failed,
						0
					),
				},
			});
		} catch (error) {
			console.error("Get queue stats error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res.status(500).json({
				message: "Error retrieving queue statistics",
				error: errorMessage,
			});
		}
	});

	/**
	 * Queue management endpoints
	 *
	 * POST /api/admin/queue-control
	 * Allows pausing/resuming queues and performing maintenance
	 */
	app.post("/api/admin/queue-control", async (req: Request, res: Response) => {
		try {
			// In production, add proper authentication and admin role checking here

			const { action } = req.body;

			switch (action) {
				case "pause":
					await jobQueueManager.pauseAllQueues();
					res.json({ message: "All queues paused", action: "pause" });
					break;

				case "resume":
					await jobQueueManager.resumeAllQueues();
					res.json({ message: "All queues resumed", action: "resume" });
					break;

				case "cleanup":
					await jobQueueManager.cleanupJobs();
					res.json({ message: "Job cleanup completed", action: "cleanup" });
					break;

				default:
					return res.status(400).json({
						message: "Invalid action. Use 'pause', 'resume', or 'cleanup'",
					});
			}
		} catch (error) {
			console.error("Queue control error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res.status(500).json({
				message: "Error controlling queues",
				error: errorMessage,
			});
		}
	});

	/**
	 * Bulk upload processing endpoint
	 *
	 * POST /api/tracks/process-bulk
	 * Process multiple tracks with different priorities
	 */
	app.post("/api/tracks/process-bulk", async (req: Request, res: Response) => {
		try {
			const {
				trackIds,
				settings,
				priority = JobPriority.NORMAL,
				useOptimization = true,
			} = req.body;

			if (!Array.isArray(trackIds) || trackIds.length === 0) {
				return res.status(400).json({ message: "Invalid track IDs array" });
			}

			if (trackIds.length > 10) {
				return res
					.status(400)
					.json({ message: "Maximum 10 tracks can be processed in bulk" });
			}

			const validatedSettings = processingSettingsSchema.parse(settings);
			const jobIds: string[] = [];
			const errors: string[] = [];

			for (const trackId of trackIds) {
				try {
					const track = await storage.getAudioTrack(trackId);

					if (!track) {
						errors.push(`Track ${trackId} not found`);
						continue;
					}

					if (track.versionCount > 3) {
						errors.push(`Track ${trackId} has reached maximum version limit`);
						continue;
					}

					// Update track status
					await storage.updateAudioTrack(trackId, {
						status: (track.extendedPaths as string[])?.length
							? "regenerate"
							: "processing",
						settings: validatedSettings,
					});

					// Generate output path
					const outputBase = path.basename(
						track.originalFilename,
						path.extname(track.originalFilename)
					);
					const fileExt = path.extname(track.originalFilename);
					const version = (track.extendedPaths as string[])?.length || 0;
					const sanitizedBaseName = outputBase.replace(/[<>:"/\\|?*\0]/g, "");
					const outputFilename = `${sanitizedBaseName}_extended_v${
						version + 1
					}${fileExt}`;
					const resultDir =
						process.env.RESULTS_DIR || path.join(process.cwd(), "results");
					const outputPath = path.join(resultDir, outputFilename); // nosemgrep: javascript.express.security.audit.express-path-join-resolve-traversal.express-path-join-resolve-traversal, javascript.lang.security.audit.path-traversal.path-join-resolve-traversal.path-join-resolve-traversal

					// Add job to queue with slight delay to spread the load
					const jobId = await jobQueueManager.addAudioProcessingJob(
						trackId,
						track.originalPath,
						outputPath,
						validatedSettings,
						track.userId || 1,
						priority,
						useOptimization
					);

					jobIds.push(jobId);
				} catch (error) {
					const errorMessage =
						error instanceof Error ? error.message : "Unknown error";
					errors.push(`Track ${trackId}: ${errorMessage}`);
				}
			}

			res.status(202).json({
				message: `Bulk processing initiated: ${jobIds.length} jobs queued`,
				jobIds,
				successCount: jobIds.length,
				errorCount: errors.length,
				errors: errors.length > 0 ? errors : undefined,
				priority,
				useOptimization,
			});
		} catch (error) {
			console.error("Bulk process error:", error);
			const errorMessage =
				error instanceof Error ? error.message : "Unknown error";
			return res.status(500).json({
				message: "Error processing bulk request",
				error: errorMessage,
			});
		}
	});

	/**
	 * Enhanced track status endpoint with job information
	 *
	 * GET /api/tracks/:id/detailed-status
	 * Returns track status including active job information
	 */
	app.get(
		"/api/tracks/:id/detailed-status",
		async (req: Request, res: Response) => {
			try {
				const id = parseInt(req.params.id, 10);
				if (isNaN(id)) {
					return res.status(400).json({ message: "Invalid track ID" });
				}

				const track = await storage.getAudioTrack(id);
				if (!track) {
					return res.status(404).json({ message: "Track not found" });
				}

				// Basic track status
				const response: any = {
					trackId: id,
					status: track.status,
					versionCount: track.versionCount,
					hasExtended: (track.extendedPaths as string[])?.length > 0,
					settings: track.settings,
				};

				// If track is being processed, try to find active job
				if (track.status === "processing" || track.status === "regenerate") {
					// Note: In a production system, you'd store job IDs with tracks
					// For now, we'll indicate that processing is active
					response.processing = {
						active: true,
						message: "Track is being processed in background job queue",
						estimatedTimeRemaining: "2-5 minutes",
					};
				}

				res.json(response);
			} catch (error) {
				console.error("Get detailed status error:", error);
				const errorMessage =
					error instanceof Error ? error.message : "Unknown error";
				return res.status(500).json({
					message: "Error retrieving detailed status",
					error: errorMessage,
				});
			}
		}
	);

	/**
	 * Job queue health check endpoint
	 *
	 * GET /api/health/job-queue
	 * Returns health status of the job queue system
	 */
	app.get("/api/health/job-queue", async (req: Request, res: Response) => {
		try {
			const stats = await jobQueueManager.getQueueStats();

			// Calculate health metrics
			const totalJobs = Object.values(stats).reduce(
				(sum: number, queue: any) => sum + queue.total,
				0
			);
			const failedJobs = Object.values(stats).reduce(
				(sum: number, queue: any) => sum + queue.failed,
				0
			);
			const activeJobs = Object.values(stats).reduce(
				(sum: number, queue: any) => sum + queue.active,
				0
			);

			const failureRate = totalJobs > 0 ? (failedJobs / totalJobs) * 100 : 0;
			const isHealthy = failureRate < 10 && activeJobs < 20; // Configurable thresholds

			res.json({
				status: isHealthy ? "healthy" : "degraded",
				timestamp: new Date().toISOString(),
				metrics: {
					totalJobs,
					activeJobs,
					failedJobs,
					failureRate: `${failureRate.toFixed(2)}%`,
				},
				queues: stats,
				thresholds: {
					maxFailureRate: "10%",
					maxActiveJobs: 20,
				},
			});
		} catch (error) {
			console.error("Job queue health check error:", error);
			res.status(503).json({
				status: "unhealthy",
				timestamp: new Date().toISOString(),
				error: error instanceof Error ? error.message : "Unknown error",
			});
		}
	});
}

export default setupJobQueueRoutes;
