import { GeoJsonMultiPolygonGeometry, GeoJsonPolygonGeometry } from '../types';

const STORAGE_KEY = 'oiltrace.detections.v1';
const MAX_STORED_DETECTIONS = 25;

export interface StoredDetectionResult {
  detectionId: string;
  createdAt: string;
  imageName: string;
  modelName: string;
  inferenceMode: string;
  isDemo: boolean;
  spillDetected: boolean;
  confidence: number;
  confidenceIsCalibrated: boolean;
  maskAreaRatio: number;
  areaKm2: number | null;
  location: { latitude: number; longitude: number } | null;
  groundSampleDistanceM: number | null;
  geometry: GeoJsonMultiPolygonGeometry | null;
  imageFootprint: GeoJsonPolygonGeometry | null;
  maskPngDataUrl: string;
}

export function loadDetectionHistory(): StoredDetectionResult[] {
  try {
    const serialized = window.localStorage.getItem(STORAGE_KEY);
    const parsed = serialized ? JSON.parse(serialized) as StoredDetectionResult[] : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveDetectionResult(result: StoredDetectionResult): StoredDetectionResult[] {
  const updated = [result, ...loadDetectionHistory().filter((entry) => entry.detectionId !== result.detectionId)]
    .slice(0, MAX_STORED_DETECTIONS);
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  } catch (error) {
    console.warn('Unable to persist OilTrace detection history in this browser.', error);
  }
  return updated;
}