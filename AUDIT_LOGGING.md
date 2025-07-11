<!-- @format -->

# Comprehensive Audit Logging Implementation

## Overview

This document describes the comprehensive audit logging system implemented in the DJ Mix Extender application to track security events, user activities, and system operations for compliance and security monitoring.

## Features Implemented

### 1. Audit Configuration

- **Environment-Based Configuration**: Flexible configuration through environment variables
- **Multiple Output Options**: Console logging, file logging, or both
- **Retention Management**: Configurable log retention period
- **Development vs Production**: Different logging behaviors based on environment

### 2. Audit Event Categories

#### Authentication & Authorization Events

- `AUTH_SUCCESS`: Successful authentication attempts
- `AUTH_FAILURE`: Failed authentication attempts
- `UNAUTHORIZED_ACCESS`: Attempts to access restricted resources

#### File Operation Events

- `FILE_UPLOAD`: File upload attempts (success/failure)
- `FILE_UPLOAD_REJECTED`: File uploads rejected due to validation
- `FILE_ACCESS`: File access events (streaming, downloading)
- `FILE_ACCESS_DENIED`: Denied file access attempts
- `FILE_DOWNLOAD`: File download events
- `FILE_DELETE`: File deletion operations

#### Audio Processing Events

- `AUDIO_PROCESSING_START`: Start of audio processing operations
- `AUDIO_PROCESSING_SUCCESS`: Successful completion of processing
- `AUDIO_PROCESSING_FAILURE`: Failed processing attempts

#### Security Violation Events

- `PATH_TRAVERSAL_ATTEMPT`: Attempted path traversal attacks
- `INVALID_FILE_TYPE`: Invalid file type uploads
- `FILE_SIZE_VIOLATION`: File size limit violations
- `MALICIOUS_FILENAME`: Malicious filename detection
- `RATE_LIMIT_EXCEEDED`: Rate limiting violations
- `SUSPICIOUS_REQUEST`: Suspicious request patterns

#### System Events

- `SYSTEM_ERROR`: System-level errors
- `CONFIGURATION_CHANGE`: Configuration modifications
- `SERVICE_START`: Application startup
- `SERVICE_STOP`: Application shutdown

### 3. Severity Levels

- **LOW**: Normal operations, successful events
- **MEDIUM**: Warning-level events, failed operations
- **HIGH**: Security violations, unauthorized access
- **CRITICAL**: Critical security breaches, system failures

### 4. Audit Event Structure

```json
{
	"timestamp": "2024-01-15T10:30:45.123Z",
	"eventType": "FILE_UPLOAD",
	"severity": "LOW",
	"outcome": "SUCCESS",
	"userId": 1,
	"ipAddress": "192.168.1.100",
	"userAgent": "Mozilla/5.0...",
	"resource": "/api/tracks/upload",
	"action": "POST",
	"requestId": "abc123def456",
	"details": {
		"filename": "song.mp3",
		"fileSize": 5242880,
		"trackId": 123
	}
}
```

### 5. Security Features

#### Data Sanitization

- **Sensitive Field Masking**: Passwords, tokens, secrets automatically redacted
- **Path Security**: File paths sanitized to prevent information leakage
- **String Truncation**: Long strings truncated to prevent log bombing
- **JSON Sanitization**: Deep sanitization of complex objects

#### Client Information Extraction

- **IP Address Detection**: Supports X-Forwarded-For, X-Real-IP headers
- **User Agent Logging**: Browser/client identification
- **Request ID Generation**: Unique identifier for request correlation

### 6. File-Based Logging

#### Log File Security

- **Secure Permissions**: Log files created with restrictive permissions (0o640)
- **Directory Security**: Log directory created with secure permissions (0o750)
- **Atomic Writes**: Append-only operations for data integrity

#### Log Rotation

- **Age-Based Rotation**: Automatic rotation based on retention period
- **Backup Creation**: Old logs archived with timestamp
- **Cleanup Management**: Automatic cleanup of expired logs

### 7. Implementation Details

#### Audit Logger Class

```typescript
class AuditLogger {
	// Singleton pattern for consistent logging
	private static instance: AuditLogger;

	// Core logging method
	public async log(
		eventType: AuditEventType,
		severity: AuditSeverity,
		outcome: "SUCCESS" | "FAILURE",
		details: Record<string, any>,
		req?: Request,
		userId?: string | number
	): Promise<void>;

	// Convenience methods for common events
	public async logFileUpload(req, filename, fileSize, outcome, reason?);
	public async logFileAccess(req, filePath, outcome, reason?);
	public async logSecurityViolation(req, violationType, details);
	public async logProcessingEvent(trackId, eventType, outcome, details);
	public async logSystemEvent(eventType, details);
}
```

#### Environment Configuration

```bash
# Enable/disable audit logging
AUDIT_LOGGING_ENABLED=true

# Output destinations
AUDIT_LOG_TO_FILE=true
AUDIT_LOG_TO_CONSOLE=true

# File configuration
AUDIT_LOG_FILE=./logs/audit.log
AUDIT_LOG_RETENTION_DAYS=90

# Detail level
AUDIT_INCLUDE_REQUEST_DETAILS=true
```

### 8. Route-Level Integration

#### Upload Route Security Auditing

- Failed upload attempts logged with details
- Path traversal attempts detected and logged
- File validation failures captured
- Successful uploads tracked with metadata

#### File Access Auditing

- All file streaming attempts logged
- Download operations tracked
- Unauthorized access attempts recorded
- Path validation failures captured

#### Processing Operation Auditing

- Processing start events logged
- Success/failure outcomes tracked
- Invalid parameter attempts recorded
- Version limit violations captured

### 9. Security Violation Detection

#### Path Traversal Detection

```typescript
// Detects and logs path traversal attempts
if (!validateAndSanitizePath(filePath)) {
	await auditLogger.logSecurityViolation(
		req,
		AuditEventType.PATH_TRAVERSAL_ATTEMPT,
		{
			attemptedPath: filePath,
			reason: "File path validation failed",
		}
	);
}
```

#### Suspicious Request Detection

```typescript
// Logs suspicious parameter values
if (isNaN(trackId)) {
	await auditLogger.logSecurityViolation(
		req,
		AuditEventType.SUSPICIOUS_REQUEST,
		{
			reason: "Invalid track ID parameter",
			providedId: req.params.id,
		}
	);
}
```

### 10. Compliance Features

#### Audit Trail Integrity

- **Immutable Logs**: Append-only log files
- **Timestamp Precision**: ISO 8601 timestamps with milliseconds
- **Request Correlation**: Unique request IDs for event correlation
- **Complete Coverage**: All security-relevant events captured

#### Privacy Protection

- **Data Minimization**: Only necessary data logged
- **Sensitive Data Protection**: Automatic redaction of sensitive fields
- **User Privacy**: User-identifiable information limited and protected

### 11. Monitoring and Alerting

#### Log Analysis Support

- **Structured JSON**: Machine-readable log format
- **Consistent Schema**: Standardized event structure
- **Correlation IDs**: Request tracking across operations
- **Severity Classification**: Priority-based event classification

#### Security Monitoring

- **Real-time Detection**: Immediate logging of security events
- **Pattern Recognition**: Consistent event types for pattern analysis
- **Threat Detection**: High-severity events for immediate attention
- **Compliance Reporting**: Structured data for compliance reports

### 12. Performance Considerations

#### Asynchronous Logging

- **Non-blocking Operations**: Logging doesn't block request processing
- **Error Handling**: Failed logging operations don't affect application
- **Resource Management**: Efficient file I/O operations

#### Log Volume Management

- **Selective Logging**: Configurable event types
- **Data Truncation**: Automatic truncation of large data
- **Rotation Policies**: Automatic log rotation to manage disk space

### 13. Production Deployment

#### Security Configuration

```bash
# Production settings
NODE_ENV=production
AUDIT_LOGGING_ENABLED=true
AUDIT_LOG_TO_FILE=true
AUDIT_LOG_TO_CONSOLE=false
AUDIT_INCLUDE_REQUEST_DETAILS=false
AUDIT_LOG_RETENTION_DAYS=365
```

#### Log File Management

- Ensure log directory exists with secure permissions
- Configure log rotation (logrotate or similar)
- Set up log monitoring and alerting
- Implement log backup and archival

### 14. Integration Examples

#### Custom Security Events

```typescript
// Log custom security violations
await auditLogger.logSecurityViolation(
	req,
	AuditEventType.RATE_LIMIT_EXCEEDED,
	{
		clientIP: req.ip,
		requestCount: 100,
		timeWindow: "1 minute",
	}
);
```

#### System Events

```typescript
// Log system configuration changes
await auditLogger.logSystemEvent(AuditEventType.CONFIGURATION_CHANGE, {
	setting: "MAX_FILE_SIZE",
	oldValue: "15MB",
	newValue: "20MB",
	changedBy: "admin",
});
```

### 15. Benefits

#### Security Benefits

- **Attack Detection**: Early detection of security threats
- **Incident Response**: Detailed audit trail for investigations
- **Compliance**: Meeting regulatory requirements
- **Forensics**: Complete event history for analysis

#### Operational Benefits

- **Debugging**: Detailed error tracking and analysis
- **Performance Monitoring**: Request patterns and volumes
- **User Behavior**: Understanding usage patterns
- **System Health**: Monitoring system events and errors

## Conclusion

The comprehensive audit logging system provides complete visibility into security events, user activities, and system operations while maintaining security, privacy, and performance standards. The system is production-ready with configurable retention, secure log handling, and integration with monitoring systems.
