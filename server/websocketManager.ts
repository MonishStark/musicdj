/**
 * WebSocket Integration for Real-time Job Progress Updates
 *
 * This module provides real-time communication between the job queue system and
 * the frontend for live progress tracking and notifications.
 *
 * Features:
 * - Real-time job progress updates
 * - User-specific notification rooms
 * - Admin monitoring dashboard
 * - Connection management and authentication
 * - Error handling and reconnection logic
 *
 * @format
 */

import { Server as SocketIOServer } from "socket.io";
import { Server as HTTPServer } from "http";
import { jobQueueManager } from "./jobQueueSimple";
import { websocketCorsOptions } from "./cors-config";

// Utility function to sanitize user input for logging
function sanitizeForLog(input: any): string {
	if (typeof input !== "string") {
		input = String(input);
	}
	// Remove newlines, carriage returns, and control characters that could be used for log injection
	return input
		.replace(/[\r\n\t\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
		.substring(0, 1000);
}

/**
 * WebSocket Event Types
 */
export interface JobUpdateEvent {
	jobId: string;
	trackId: number;
	userId: number;
	status:
		| "waiting"
		| "active"
		| "completed"
		| "failed"
		| "progress"
		| "stalled";
	progress?: any;
	data?: any;
	timestamp: string;
}

export interface NotificationEvent {
	type: "processing_complete" | "processing_failed" | "analysis_complete";
	trackId: number;
	message: string;
	timestamp: string;
}

export interface QueueStatsEvent {
	audioProcessing: any;
	audioAnalysis: any;
	fileCleanup: any;
	notifications: any;
	timestamp: string;
}

/**
 * WebSocket Server Manager
 *
 * Manages WebSocket connections for real-time job progress tracking
 * and user notifications.
 */
export class WebSocketManager {
	private io: SocketIOServer;
	private connectedUsers: Map<string, number> = new Map(); // socketId -> userId
	private userSockets: Map<number, Set<string>> = new Map(); // userId -> Set of socketIds

	constructor(httpServer: HTTPServer) {
		this.io = new SocketIOServer(httpServer, {
			cors: websocketCorsOptions,
			transports: ["websocket", "polling"],
			pingTimeout: 60000,
			pingInterval: 25000,
		});

		this.setupEventHandlers();
		this.setupPeriodicUpdates();

		// Inject Socket.IO instance into job queue manager
		jobQueueManager.setSocketIo(this.io);

		console.log("WebSocket server initialized");
	}

	/**
	 * Set up WebSocket event handlers
	 */
	private setupEventHandlers() {
		this.io.on("connection", (socket) => {
			console.log(`Client connected: ${socket.id}`);

			// Handle user authentication/identification
			socket.on(
				"authenticate",
				(data: { userId: number; isAdmin?: boolean }) => {
					this.authenticateUser(socket, data.userId, data.isAdmin || false);
				}
			);

			// Handle job status requests
			socket.on("get-job-status", async (data: { jobId: string }) => {
				try {
					const status = await jobQueueManager.getJobStatus(data.jobId);
					socket.emit("job-status", { jobId: data.jobId, status });
				} catch (error) {
					socket.emit("error", {
						type: "job-status-error",
						message: "Failed to get job status",
						error: error instanceof Error ? error.message : "Unknown error",
					});
				}
			});

			// Handle job cancellation requests
			socket.on("cancel-job", async (data: { jobId: string }) => {
				try {
					const userId = this.connectedUsers.get(socket.id);
					if (!userId) {
						socket.emit("error", {
							type: "auth-error",
							message: "User not authenticated",
						});
						return;
					}

					const cancelled = await jobQueueManager.cancelJob(data.jobId);
					socket.emit("job-cancelled", { jobId: data.jobId, cancelled });

					if (cancelled) {
						// Notify user about cancellation
						this.io.to(`user_${userId}`).emit("notification", {
							type: "job_cancelled",
							message: "Job cancelled successfully",
							timestamp: new Date().toISOString(),
						});
					}
				} catch (error) {
					socket.emit("error", {
						type: "cancel-error",
						message: "Failed to cancel job",
						error: error instanceof Error ? error.message : "Unknown error",
					});
				}
			});

			// Handle queue statistics requests (admin only)
			socket.on("get-queue-stats", async () => {
				try {
					const userId = this.connectedUsers.get(socket.id);
					if (!userId) {
						socket.emit("error", {
							type: "auth-error",
							message: "User not authenticated",
						});
						return;
					}

					// For demo purposes, allow any authenticated user to see stats
					// In production, implement proper admin role checking
					const stats = await jobQueueManager.getQueueStats();
					socket.emit("queue-stats", {
						...stats,
						timestamp: new Date().toISOString(),
					});
				} catch (error) {
					socket.emit("error", {
						type: "stats-error",
						message: "Failed to get queue statistics",
						error: error instanceof Error ? error.message : "Unknown error",
					});
				}
			});

			// Handle queue management requests (admin only)
			socket.on("pause-queues", async () => {
				try {
					// In production, implement proper admin role checking
					await jobQueueManager.pauseAllQueues();
					this.io.to("admin").emit("queues-paused", {
						timestamp: new Date().toISOString(),
					});
				} catch (error) {
					socket.emit("error", {
						type: "pause-error",
						message: "Failed to pause queues",
						error: error instanceof Error ? error.message : "Unknown error",
					});
				}
			});

			socket.on("resume-queues", async () => {
				try {
					// In production, implement proper admin role checking
					await jobQueueManager.resumeAllQueues();
					this.io.to("admin").emit("queues-resumed", {
						timestamp: new Date().toISOString(),
					});
				} catch (error) {
					socket.emit("error", {
						type: "resume-error",
						message: "Failed to resume queues",
						error: error instanceof Error ? error.message : "Unknown error",
					});
				}
			});

			// Handle client heartbeat
			socket.on("ping", () => {
				socket.emit("pong", { timestamp: new Date().toISOString() });
			});

			// Handle disconnection
			socket.on("disconnect", (reason) => {
				this.handleDisconnection(socket, reason);
			});

			// Send welcome message
			socket.emit("connected", {
				socketId: socket.id,
				timestamp: new Date().toISOString(),
				message: "Connected to job queue server",
			});
		});
	}

	/**
	 * Authenticate and register user
	 */
	private authenticateUser(socket: any, userId: number, isAdmin = false) {
		// Remove any existing association for this socket
		this.handleDisconnection(socket, "re-authentication");

		// Register new association
		this.connectedUsers.set(socket.id, userId);

		if (!this.userSockets.has(userId)) {
			this.userSockets.set(userId, new Set());
		}
		this.userSockets.get(userId)?.add(socket.id);

		// Join user-specific room
		socket.join(`user_${userId}`);

		// Join admin room if admin
		if (isAdmin) {
			socket.join("admin");
		}

		console.log(
			`User ${userId} authenticated with socket ${socket.id}${
				isAdmin ? " (admin)" : ""
			}`
		);

		// Send authentication confirmation
		socket.emit("authenticated", {
			userId,
			isAdmin,
			timestamp: new Date().toISOString(),
			message: "Successfully authenticated",
		});

		// Send current queue stats to admin users
		if (isAdmin) {
			this.sendQueueStatsToSocket(socket);
		}
	}

	/**
	 * Handle client disconnection
	 */
	private handleDisconnection(socket: any, reason: string) {
		const userId = this.connectedUsers.get(socket.id);

		if (userId) {
			// Remove socket from user's socket set
			const userSocketSet = this.userSockets.get(userId);
			if (userSocketSet) {
				userSocketSet.delete(socket.id);

				// If no more sockets for this user, remove the user entry
				if (userSocketSet.size === 0) {
					this.userSockets.delete(userId);
				}
			}

			console.log(
				`User ${sanitizeForLog(userId)} disconnected: ${
					socket.id
				} (${sanitizeForLog(reason)})`
			);
		} else {
			console.log(
				`Anonymous client disconnected: ${socket.id} (${sanitizeForLog(
					reason
				)})`
			);
		}

		// Remove socket from connected users
		this.connectedUsers.delete(socket.id);
	}

	/**
	 * Send queue statistics to a specific socket
	 */
	private async sendQueueStatsToSocket(socket: any) {
		try {
			const stats = await jobQueueManager.getQueueStats();
			socket.emit("queue-stats", {
				...stats,
				timestamp: new Date().toISOString(),
			});
		} catch (error) {
			console.error("Error sending queue stats:", error);
		}
	}

	/**
	 * Set up periodic updates for monitoring
	 */
	private setupPeriodicUpdates() {
		// Send queue statistics to admin users every 30 seconds
		setInterval(async () => {
			try {
				const stats = await jobQueueManager.getQueueStats();
				this.io.to("admin").emit("queue-stats", {
					...stats,
					timestamp: new Date().toISOString(),
				});
			} catch (error) {
				console.error("Error in periodic stats update:", error);
			}
		}, 30000);

		// Cleanup old connections every 5 minutes
		setInterval(() => {
			this.cleanupStaleConnections();
		}, 5 * 60 * 1000);
	}

	/**
	 * Clean up stale connections
	 */
	private cleanupStaleConnections() {
		let cleanedCount = 0;

		for (const [socketId, userId] of this.connectedUsers.entries()) {
			const socket = this.io.sockets.sockets.get(socketId);

			if (!socket || socket.disconnected) {
				// Socket is stale, clean it up
				this.connectedUsers.delete(socketId);

				const userSocketSet = this.userSockets.get(userId);
				if (userSocketSet) {
					userSocketSet.delete(socketId);
					if (userSocketSet.size === 0) {
						this.userSockets.delete(userId);
					}
				}

				cleanedCount++;
			}
		}

		if (cleanedCount > 0) {
			console.log(`Cleaned up ${cleanedCount} stale socket connections`);
		}
	}

	/**
	 * Broadcast notification to specific user
	 */
	broadcastToUser(userId: number, event: string, data: any) {
		this.io.to(`user_${userId}`).emit(event, data);
	}

	/**
	 * Broadcast notification to all admin users
	 */
	broadcastToAdmins(event: string, data: any) {
		this.io.to("admin").emit(event, data);
	}

	/**
	 * Broadcast notification to all connected users
	 */
	broadcastToAll(event: string, data: any) {
		this.io.emit(event, data);
	}

	/**
	 * Get connection statistics
	 */
	getConnectionStats() {
		return {
			totalConnections: this.connectedUsers.size,
			totalUsers: this.userSockets.size,
			userConnections: Array.from(this.userSockets.entries()).map(
				([userId, sockets]) => ({
					userId,
					socketCount: sockets.size,
				})
			),
			adminConnections: this.io.sockets.adapter.rooms.get("admin")?.size || 0,
		};
	}

	/**
	 * Graceful shutdown
	 */
	async shutdown() {
		console.log("Shutting down WebSocket server...");

		// Notify all connected clients about shutdown
		this.io.emit("server-shutdown", {
			message: "Server is shutting down",
			timestamp: new Date().toISOString(),
		});

		// Close all connections
		this.io.close();

		console.log("WebSocket server shutdown completed");
	}
}

export default WebSocketManager;
