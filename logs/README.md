<!-- @format -->

# Logs Directory

This directory is used for storing audit logs and other application logs.

## Security Notes

- Log files contain sensitive information and should be protected
- Ensure proper file permissions are set (0o640 for files, 0o750 for directories)
- Log files are automatically managed by the audit logging system
- Old logs are rotated based on the retention policy

## Configuration

Log configuration is managed through environment variables:

- `AUDIT_LOG_FILE`: Path to the audit log file (default: ./logs/audit.log)
- `AUDIT_LOG_RETENTION_DAYS`: How long to keep logs (default: 90 days)
- `AUDIT_LOG_TO_FILE`: Enable file logging (default: true in production)

## Files

- `audit.log`: Main audit log file (JSON format)
- `audit.log.*.old`: Rotated/archived log files
- Other application-specific log files as needed
