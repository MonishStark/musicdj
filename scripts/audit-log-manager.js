#!/usr/bin/env node
/**
 * Audit Log Management Utility
 *
 * This utility provides commands for managing audit logs, including:
 * - Log rotation
 * - Log analysis
 * - Security event monitoring
 * - Log cleanup
 *
 * @format
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const AUDIT_LOG_PATH =
	process.env.AUDIT_LOG_FILE || path.join(__dirname, "..", "logs", "audit.log");
const RETENTION_DAYS = parseInt(
	process.env.AUDIT_LOG_RETENTION_DAYS || "90",
	10
);

class AuditLogManager {
	/**
	 * Rotate log files based on retention policy
	 */
	static async rotateLogs() {
		try {
			if (!fs.existsSync(AUDIT_LOG_PATH)) {
				console.log("No audit log file found to rotate");
				return;
			}

			const stats = await fs.promises.stat(AUDIT_LOG_PATH);
			const fileAge = Date.now() - stats.mtime.getTime();
			const maxAge = RETENTION_DAYS * 24 * 60 * 60 * 1000;

			if (fileAge > maxAge) {
				const backupPath = `${AUDIT_LOG_PATH}.${Date.now()}.old`;
				await fs.promises.rename(AUDIT_LOG_PATH, backupPath);
				console.log(`Log rotated: ${path.basename(backupPath)}`);
			} else {
				console.log("Log rotation not needed yet");
			}
		} catch (error) {
			console.error("Failed to rotate logs:", error);
		}
	}

	/**
	 * Clean up old log files
	 */
	static async cleanupOldLogs() {
		try {
			const logDir = path.dirname(AUDIT_LOG_PATH);
			const files = await fs.promises.readdir(logDir);
			const maxAge = RETENTION_DAYS * 24 * 60 * 60 * 1000;
			let cleanedCount = 0;

			for (const file of files) {
				if (file.endsWith(".old")) {
					const filePath = path.join(logDir, file);
					const stats = await fs.promises.stat(filePath);
					const fileAge = Date.now() - stats.mtime.getTime();

					if (fileAge > maxAge) {
						await fs.promises.unlink(filePath);
						cleanedCount++;
						console.log(`Deleted old log: ${file}`);
					}
				}
			}

			console.log(`Cleanup complete: ${cleanedCount} old log files deleted`);
		} catch (error) {
			console.error("Failed to cleanup old logs:", error);
		}
	}

	/**
	 * Analyze audit logs for security events
	 */
	static async analyzeSecurityEvents(hours = 24) {
		try {
			if (!fs.existsSync(AUDIT_LOG_PATH)) {
				console.log("No audit log file found");
				return;
			}

			const content = await fs.promises.readFile(AUDIT_LOG_PATH, "utf-8");
			const lines = content
				.trim()
				.split("\n")
				.filter((line) => line.trim());
			const cutoffTime = Date.now() - hours * 60 * 60 * 1000;

			const securityEvents = [];
			const summary = {
				totalEvents: 0,
				securityViolations: 0,
				failedAccess: 0,
				suspiciousRequests: 0,
				pathTraversalAttempts: 0,
				uniqueIPs: new Set(),
			};

			for (const line of lines) {
				try {
					const event = JSON.parse(line);
					const eventTime = new Date(event.timestamp).getTime();

					if (eventTime >= cutoffTime) {
						summary.totalEvents++;

						if (event.ipAddress) {
							summary.uniqueIPs.add(event.ipAddress);
						}

						// Classify security events
						if (event.severity === "HIGH" || event.severity === "CRITICAL") {
							securityEvents.push(event);
							summary.securityViolations++;
						}

						if (
							event.eventType.includes("ACCESS_DENIED") ||
							event.outcome === "FAILURE"
						) {
							summary.failedAccess++;
						}

						if (event.eventType === "SUSPICIOUS_REQUEST") {
							summary.suspiciousRequests++;
						}

						if (event.eventType === "PATH_TRAVERSAL_ATTEMPT") {
							summary.pathTraversalAttempts++;
						}
					}
				} catch (parseError) {
					// Skip malformed lines
				}
			}

			// Display summary
			console.log(`\n=== Security Analysis (Last ${hours} hours) ===`);
			console.log(`Total Events: ${summary.totalEvents}`);
			console.log(`Security Violations: ${summary.securityViolations}`);
			console.log(`Failed Access Attempts: ${summary.failedAccess}`);
			console.log(`Suspicious Requests: ${summary.suspiciousRequests}`);
			console.log(`Path Traversal Attempts: ${summary.pathTraversalAttempts}`);
			console.log(`Unique IP Addresses: ${summary.uniqueIPs.size}`);

			// Display critical events
			if (securityEvents.length > 0) {
				console.log("\n=== Critical Security Events ===");
				securityEvents.slice(0, 10).forEach((event, index) => {
					console.log(`\n${index + 1}. ${event.eventType} (${event.severity})`);
					console.log(`   Time: ${event.timestamp}`);
					console.log(`   IP: ${event.ipAddress || "unknown"}`);
					console.log(`   Details: ${JSON.stringify(event.details, null, 2)}`);
				});
			}
		} catch (error) {
			console.error("Failed to analyze security events:", error);
		}
	}

	/**
	 * Monitor logs in real-time
	 */
	static async monitorLogs() {
		try {
			console.log("Monitoring audit logs (Press Ctrl+C to stop)...\n");

			if (!fs.existsSync(AUDIT_LOG_PATH)) {
				console.log("Waiting for audit log file to be created...");
			}

			let lastPosition = 0;
			if (fs.existsSync(AUDIT_LOG_PATH)) {
				lastPosition = fs.statSync(AUDIT_LOG_PATH).size;
			}

			const watcher = fs.watchFile(
				AUDIT_LOG_PATH,
				{ interval: 1000 },
				(curr, prev) => {
					if (curr.size > lastPosition) {
						const stream = fs.createReadStream(AUDIT_LOG_PATH, {
							start: lastPosition,
							end: curr.size,
						});

						let data = "";
						stream.on("data", (chunk) => {
							data += chunk.toString();
						});

						stream.on("end", () => {
							const lines = data.split("\n").filter((line) => line.trim());
							for (const line of lines) {
								try {
									const event = JSON.parse(line);
									this.displayEvent(event);
								} catch (error) {
									// Skip malformed lines
								}
							}
							lastPosition = curr.size;
						});
					}
				}
			);

			// Handle Ctrl+C
			process.on("SIGINT", () => {
				fs.unwatchFile(AUDIT_LOG_PATH);
				console.log("\nMonitoring stopped");
				process.exit(0);
			});
		} catch (error) {
			console.error("Failed to monitor logs:", error);
		}
	}

	/**
	 * Display a formatted audit event
	 */
	static displayEvent(event) {
		const timestamp = new Date(event.timestamp).toLocaleString();
		const severity = event.severity.padEnd(8);
		const outcome = event.outcome === "SUCCESS" ? "✓" : "✗";
		const ip = event.ipAddress ? ` [${event.ipAddress}]` : "";

		let color = "";
		if (event.severity === "CRITICAL") color = "\x1b[31m"; // Red
		else if (event.severity === "HIGH") color = "\x1b[33m"; // Yellow
		else if (event.outcome === "FAILURE") color = "\x1b[91m"; // Light Red
		else color = "\x1b[32m"; // Green

		console.log(
			`${color}${timestamp} [${severity}] ${outcome} ${event.eventType}${ip}\x1b[0m`
		);

		if (event.details && Object.keys(event.details).length > 0) {
			console.log(`   ${JSON.stringify(event.details)}`);
		}
	}

	/**
	 * Display log statistics
	 */
	static async showStats() {
		try {
			if (!fs.existsSync(AUDIT_LOG_PATH)) {
				console.log("No audit log file found");
				return;
			}

			const stats = await fs.promises.stat(AUDIT_LOG_PATH);
			const content = await fs.promises.readFile(AUDIT_LOG_PATH, "utf-8");
			const lines = content
				.trim()
				.split("\n")
				.filter((line) => line.trim());

			console.log("=== Audit Log Statistics ===");
			console.log(`File Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
			console.log(`Total Events: ${lines.length}`);
			console.log(`Created: ${stats.birthtime.toLocaleString()}`);
			console.log(`Modified: ${stats.mtime.toLocaleString()}`);

			// Count events by type
			const eventTypes = {};
			const severities = {};

			for (const line of lines) {
				try {
					const event = JSON.parse(line);
					eventTypes[event.eventType] = (eventTypes[event.eventType] || 0) + 1;
					severities[event.severity] = (severities[event.severity] || 0) + 1;
				} catch (error) {
					// Skip malformed lines
				}
			}

			console.log("\n=== Event Types ===");
			Object.entries(eventTypes)
				.sort(([, a], [, b]) => b - a)
				.slice(0, 10)
				.forEach(([type, count]) => {
					console.log(`${type}: ${count}`);
				});

			console.log("\n=== Severity Distribution ===");
			Object.entries(severities).forEach(([severity, count]) => {
				console.log(`${severity}: ${count}`);
			});
		} catch (error) {
			console.error("Failed to show statistics:", error);
		}
	}
}

// CLI Interface
const command = process.argv[2];
const args = process.argv.slice(3);

switch (command) {
	case "rotate":
		AuditLogManager.rotateLogs();
		break;

	case "cleanup":
		AuditLogManager.cleanupOldLogs();
		break;

	case "analyze":
		const hours = parseInt(args[0]) || 24;
		AuditLogManager.analyzeSecurityEvents(hours);
		break;

	case "monitor":
		AuditLogManager.monitorLogs();
		break;

	case "stats":
		AuditLogManager.showStats();
		break;

	default:
		console.log("Audit Log Management Utility");
		console.log("");
		console.log("Usage: node audit-log-manager.js <command> [options]");
		console.log("");
		console.log("Commands:");
		console.log(
			"  rotate              Rotate log files based on retention policy"
		);
		console.log("  cleanup             Clean up old log files");
		console.log(
			"  analyze [hours]     Analyze security events (default: 24 hours)"
		);
		console.log("  monitor             Monitor logs in real-time");
		console.log("  stats               Show log file statistics");
		console.log("");
		console.log("Examples:");
		console.log(
			"  node audit-log-manager.js analyze 48   # Analyze last 48 hours"
		);
		console.log(
			"  node audit-log-manager.js monitor      # Real-time monitoring"
		);
		console.log("  node audit-log-manager.js cleanup      # Clean old logs");
		break;
}
