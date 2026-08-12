/**
 * KishoLens API Configuration Utility
 * Centralizes backend API URL resolution across development and production environments.
 */

export const API_BASE_URL = (typeof process !== "undefined" && process.env && process.env.PUBLIC_API_URL)
  || (import.meta && import.meta.env && import.meta.env.PUBLIC_API_URL)
  || "http://localhost:8000";

export function getApiEndpoint(path) {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}
