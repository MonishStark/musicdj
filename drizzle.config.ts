/** @format */

import { defineConfig } from "drizzle-kit";

// Validate required environment variables
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
	throw new Error("DATABASE_URL environment variable is required");
}

export default defineConfig({
	out: "./migrations",
	schema: "./shared/schema.ts",
	dialect: "postgresql",
	dbCredentials: {
		url: databaseUrl,
	},
});
