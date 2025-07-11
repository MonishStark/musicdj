/** @format */

// Test file to verify path sanitization security
import "dotenv/config";
import path from "path";
import fs from "fs";

// Security: Define allowed directories for file operations
const uploadsDir = path.join(process.cwd(), "uploads");
const resultDir = path.join(process.cwd(), "results");
const ALLOWED_DIRECTORIES = [path.resolve(uploadsDir), path.resolve(resultDir)];

// Security: Allowed file extensions
const ALLOWED_EXTENSIONS = [".mp3", ".wav", ".flac", ".aiff"];

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

console.log("Testing path sanitization security...\n");

// Test cases
const testCases = [
	// Valid paths
	{
		path: path.join(uploadsDir, "test.mp3"),
		expected: "valid",
		description: "Valid upload path",
	},
	{
		path: path.join(resultDir, "output.wav"),
		expected: "valid",
		description: "Valid result path",
	},

	// Path traversal attempts
	{
		path: "../../../etc/passwd",
		expected: "invalid",
		description: "Path traversal attempt",
	},
	{
		path: path.join(uploadsDir, "../../../etc/passwd"),
		expected: "invalid",
		description: "Path traversal from uploads",
	},
	{
		path: uploadsDir + "/../../../etc/passwd",
		expected: "invalid",
		description: "Path traversal with concatenation",
	},

	// Invalid extensions
	{
		path: path.join(uploadsDir, "script.js"),
		expected: "invalid",
		description: "JavaScript file",
	},
	{
		path: path.join(uploadsDir, "executable.exe"),
		expected: "invalid",
		description: "Executable file",
	},

	// Invalid directories
	{ path: "/etc/passwd", expected: "invalid", description: "System file" },
	{
		path: "C:\\Windows\\System32\\notepad.exe",
		expected: "invalid",
		description: "Windows system file",
	},

	// Special characters
	{
		path: path.join(uploadsDir, "file~backup.mp3"),
		expected: "invalid",
		description: "File with tilde",
	},
	{
		path: path.join(uploadsDir, "normal_file.mp3"),
		expected: "valid",
		description: "Normal filename",
	},
];

let passedTests = 0;
let totalTests = testCases.length;

testCases.forEach((testCase, index) => {
	const result = validateAndSanitizePath(testCase.path);
	const isValid = result !== null;
	const expectedValid = testCase.expected === "valid";
	const passed = isValid === expectedValid;

	console.log(`Test ${index + 1}: ${testCase.description}`);
	console.log(`  Input: ${testCase.path}`);
	console.log(`  Expected: ${testCase.expected}`);
	console.log(`  Actual: ${isValid ? "valid" : "invalid"}`);
	console.log(`  Result: ${result || "null"}`);
	console.log(`  Status: ${passed ? "✅ PASS" : "❌ FAIL"}\n`);

	if (passed) passedTests++;
});

console.log(`\nTest Summary: ${passedTests}/${totalTests} tests passed`);

if (passedTests === totalTests) {
	console.log(
		"🎉 All security tests passed! Path sanitization is working correctly."
	);
	process.exit(0);
} else {
	console.log(
		"❌ Some security tests failed. Please review the implementation."
	);
	process.exit(1);
}
