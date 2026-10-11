import { CorrelatedVessel, MetoceanData } from '../types';
import { DriftSimulationResult } from '../types';

export const SAMPLE_METOCEAN_SOURCE = 'Sample development metocean values';
export const SAMPLE_AIS_SOURCE = 'Synthetic AIS tracks for workflow testing';
export const ATTRIBUTION_DISCLAIMER = 'Candidate scores indicate investigative relevance only. Vessel tracks are synthetic and are not evidence of responsibility.';

export function metoceanFromVectors(
  wind: { wind_u: number; wind_v: number },
  current: { u_current: number; v_current: number },
  source: string,
): MetoceanData {
  const windSpeedMs = Math.hypot(wind.wind_u, wind.wind_v);
  const windDirectionDeg = (Math.atan2(-wind.wind_u, -wind.wind_v) * 180 / Math.PI + 360) % 360;
  const currentSpeedMs = Math.hypot(current.u_current, current.v_current);
  const currentDirectionDeg = (Math.atan2(current.u_current, current.v_current) * 180 / Math.PI + 360) % 360;
  return {
    windSpeedKts: Number((windSpeedMs * 1.94384).toFixed(1)),
    windDirectionDeg: Math.round(windDirectionDeg),
    surfaceCurrentSpeedMs: Number(currentSpeedMs.toFixed(2)),
    surfaceCurrentDirDeg: Math.round(currentDirectionDeg),
    seaSurfaceTempC: 18,
    waveHeightM: 1.2,
    source,
  };
}

export interface AisTrackRecord {
  mmsi: string;
  name: string;
  vessel_type: CorrelatedVessel['vesselType'];
  timestamp: string;
  latitude: number;
  longitude: number;
  speed_knots: number;
  course_over_ground: number;
  source: string;
}

export function candidatesFromAisRecords(records: AisTrackRecord[]): CorrelatedVessel[] {
  const byMmsi = new Map<string, AisTrackRecord[]>();
  for (const record of records) {
    byMmsi.set(record.mmsi, [...(byMmsi.get(record.mmsi) || []), record]);
  }
  return [...byMmsi.entries()].map(([mmsi, track], index) => {
    const first = track[0];
    const last = track[track.length - 1];
    return {
      id: `SYNTHETIC-${mmsi}`,
      name: first.name,
      mmsi,
      imo: `SYNTHETIC-${index + 1}`,
      callsign: `SAMPLE${index + 1}`,
      flag: 'Synthetic sample',
      flagCode: 'XX',
      vesselType: first.vessel_type,
      lengthM: 100,
      beamM: 20,
      draftM: 5,
      draftChangeM: 0,
      speedKnots: last.speed_knots,
      courseOverGround: last.course_over_ground,
      distanceAtOriginNm: 0,
      timeDeviationMinutes: 0,
      attributionScore: 0,
      aisStatus: 'NORMAL',
      ownerCompany: 'Synthetic sample record',
      registeredPort: 'Sample port',
      destination: 'Sample destination',
      eta: last.timestamp,
      suspectedAction: 'Synthetic candidate used to exercise ranking only.',
      trackCoordinates: track.map((record) => ({
        lat: record.latitude,
        lon: record.longitude,
        timeUtc: record.timestamp,
        speedKts: record.speed_knots,
        courseDeg: record.course_over_ground,
      })),
    };
  });
}