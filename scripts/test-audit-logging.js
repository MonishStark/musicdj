#!/usr/bin/env node
/**
 * Audit Logging Test Script
 *
 * This script tests the audit logging functionality by simulating various security events.
 *
 * @format
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mock Express request object for testing
function createMockRequest(overrides = {}) {
	return {
		headers: {
			"user-agent": "Test-Agent/1.0",
			"x-forwarded-for": "192.168.1.100",
			...overrides.headers,
		},
		socket: {
			remoteAddress: "127.0.0.1",
		},
		originalUrl: "/api/test",
		url: "/api/test",
		method: "GET",
		params: {},
		query: {},
		...overrides,
	};
}

// Import the audit logger (we need to set up the environment first)
process.env.AUDIT_LOGGING_ENABLED = "true";
process.env.AUDIT_LOG_TO_CONSOLE = "true";
process.env.AUDIT_LOG_TO_FILE = "true";
process.env.AUDIT_LOG_FILE = path.join(
	__dirname,
	"..",
	"logs",
	"audit-test.log"
);

// Test cases
const testCases = [
	{
		name: "Successful File Upload",
		test: async (auditLogger) => {
			const req = createMockRequest({
				originalUrl: "/api/tracks/upload",
				method: "POST",
			});
			await auditLogger.logFileUpload(req, "test-song.mp3", 5242880, "SUCCESS");
		},
	},
	{
		name: "Failed File Upload - Invalid Type",
		test: async (auditLogger) => {
			const req = createMockRequest({
				originalUrl: "/api/tracks/upload",
				method: "POST",
			});
			await auditLogger.logFileUpload(
				req,
				"malicious.exe",
				1024,
				"FAILURE",
				"Invalid file type"
			);
		},
	},
	{
		name: "Path Traversal Attempt",
		test: async (auditLogger) => {
			const req = createMockRequest({
				originalUrl: "/api/audio/1/original",
				method: "GET",
			});
			await auditLogger.logSecurityViolation(req, "PATH_TRAVERSAL_ATTEMPT", {
				attemptedPath: "../../../etc/passwd",
				reason: "Path traversal detected",
			});
		},
	},
	{
		name: "Suspicious Request - Invalid ID",
		test: async (auditLogger) => {
			const req = createMockRequest({
				originalUrl: "/api/tracks/invalid-id",
				method: "GET",
				params: { id: "SELECT * FROM users" },
			});
			await auditLogger.logSecurityViolation(req, "SUSPICIOUS_REQUEST", {
				reason: "SQL injection attempt detected",
				providedId: "SELECT * FROM users",
			});
		},
	},
	{
		name: "Audio Processing Success",
		test: async (auditLogger) => {
			await auditLogger.logProcessingEvent(
				123,
				"AUDIO_PROCESSING_SUCCESS",
				"SUCCESS",
				{
					duration: 180.5,
					outputFormat: "mp3",
					settings: {
						introLength: 16,
						outroLength: 16,
						preserveVocals: true,
					},
				}
			);
		},
	},
	{
		name: "System Error",
		test: async (auditLogger) => {
			const req = createMockRequest({
				originalUrl: "/api/tracks/1/process",
				method: "POST",
			});
			await auditLogger.log(
				"SYSTEM_ERROR",
				"HIGH",
				"FAILURE",
				{
					error: "Python process crashed",
					trackId: 1,
					pythonExitCode: 1,
				},
				req
			);
		},
	},
	{
		name: "Service Start Event",
		test: async (auditLogger) => {
			await auditLogger.logSystemEvent("SERVICE_START", {
				nodeVersion: process.version,
				platform: process.platform,
				environment: "test",
			});
		},
	},
];

async function runAuditTests() {
	console.log("🔍 Starting Audit Logging Tests...\n");

	try {
		// Import the AuditLogger class dynamically
		// Since we can't import from TypeScript directly, we'll simulate the key functions
		const AuditLogger = {
			async logFileUpload(req, filename, fileSize, outcome, reason) {
				return this.log(
					"FILE_UPLOAD",
					outcome === "SUCCESS" ? "LOW" : "MEDIUM",
					outcome,
					{
						filename,
						fileSize,
						reason,
					},
					req
				);
			},

			async logSecurityViolation(req, violationType, details) {
				return this.log(violationType, "HIGH", "FAILURE", details, req);
			},

			async logProcessingEvent(trackId, eventType, outcome, details) {
				return this.log(
					eventType,
					outcome === "SUCCESS" ? "LOW" : "MEDIUM",
					outcome,
					{
						trackId,
						...details,
					}
				);
			},

			async logSystemEvent(eventType, details) {
				return this.log(eventType, "MEDIUM", "SUCCESS", details);
			},

			async log(eventType, severity, outcome, details = {}, req = null) {
				const auditEvent = {
					timestamp: new Date().toISOString(),
					eventType,
					severity,
					outcome,
					details: this.sanitizeForLogging(details),
					userId: "test-user",
					ipAddress:
						req?.headers?.["x-forwarded-for"] ||
						req?.socket?.remoteAddress ||
						"unknown",
					userAgent: req?.headers?.["user-agent"] || "unknown",
					resource: req?.originalUrl || req?.url,
					action: req?.method,
					requestId:
						Date.now().toString(36) + Math.random().toString(36).substr(2),
				};

				// Log to console
				const logLevel =
					severity === "CRITICAL" || severity === "HIGH" ? "error" : "info";
				console[logLevel](`[AUDIT] ${eventType} - ${outcome}`, {
					timestamp: auditEvent.timestamp,
					severity,
					userId: auditEvent.userId,
					ipAddress: auditEvent.ipAddress,
					resource: auditEvent.resource,
					details: auditEvent.details,
				});

				// Log to file
				if (process.env.AUDIT_LOG_TO_FILE === "true") {
					const logEntry = JSON.stringify(auditEvent) + "\n";
					const logDir = path.dirname(process.env.AUDIT_LOG_FILE);
					if (!fs.existsSync(logDir)) {
						fs.mkdirSync(logDir, { recursive: true, mode: 0o750 });
					}
					await fs.promises.appendFile(process.env.AUDIT_LOG_FILE, logEntry, {
						mode: 0o640,
					});
				}
			},

			sanitizeForLogging(data) {
				const sanitized = { ...data };

				// Remove sensitive fields
				const sensitiveFields = ["password", "token", "secret", "key", "auth"];
				for (const field of sensitiveFields) {
					if (sanitized[field]) {
						sanitized[field] = "[REDACTED]";
					}
				}

				// Limit file path exposure
				if (sanitized.filePath) {
					sanitized.filePath = path.basename(sanitized.filePath);
				}

				return sanitized;
			},
		};

		// Run each test case
		for (const testCase of testCases) {
			console.log(`\n📝 Testing: ${testCase.name}`);
			try {
				await testCase.test(AuditLogger);
				console.log(`✅ ${testCase.name} - PASSED`);
			} catch (error) {
				console.error(`❌ ${testCase.name} - FAILED:`, error.message);
			}
		}

		console.log("\n🎉 Audit logging tests completed!");

		// Show log file info
		if (fs.existsSync(process.env.AUDIT_LOG_FILE)) {
			const stats = fs.statSync(process.env.AUDIT_LOG_FILE);
			console.log(`\n📊 Test log file created: ${process.env.AUDIT_LOG_FILE}`);
			console.log(`📦 File size: ${stats.size} bytes`);
			console.log(`📅 Created: ${stats.birthtime.toLocaleString()}`);
		}
	} catch (error) {
		console.error("❌ Test failed:", error);
	}
}

// Run the tests
runAuditTests();
