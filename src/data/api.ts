import { z } from 'zod';
import { API_BASE_URL, USE_MOCK } from '@/config';
import { dams as mockDams, makeForecast, makeHistory } from './mock';
import type { Dam, Forecast, Horizon, LevelPoint } from './types';

const damSchema = z.object({ id: z.string(), name: z.string(), reservoir: z.string(), river: z.string(), location: z.string(), district: z.string(), state: z.string(), lat: z.number(), lng: z.number(), yearBuilt: z.number().optional(), purpose: z.string(), maxLevel: z.number(), minLevel: z.number(), unit: z.string(), grossStorage: z.number(), liveStorage: z.number(), currentStorage: z.number().optional(), deadStorage: z.number(), catchmentArea: z.number(), nearbyPlaces: z.array(z.string()), affectedRegions: z.array(z.string()), lastRecorded: z.object({ date: z.string(), level: z.number() }) });
const levelSchema = z.object({ date: z.string(), level: z.number() });
const forecastSchema = z.object({ damId: z.string(), asOf: z.string(), horizon: z.union([z.literal(1), z.literal(7), z.literal(14), z.literal(30)]), predictions: z.array(z.object({ date: z.string(), level: z.number(), lower: z.number().optional(), upper: z.number().optional() })), confidence: z.object({ 1: z.number(), 7: z.number(), 14: z.number(), 30: z.number() }), persistence: z.object({ 1: z.number(), 7: z.number(), 14: z.number(), 30: z.number() }), strategies: z.object({ 1: z.string().optional(), 7: z.string().optional(), 14: z.string().optional(), 30: z.string().optional() }).optional(), actuals: z.array(levelSchema).optional() });

async function readJson<T>(url: string, schema: z.ZodType<T>, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return schema.parse(await response.json());
}
const base = API_BASE_URL.replace(/\/$/, '');
const safe = (value: unknown, fallback = 0) => value == null || value === '' ? fallback : Number.isFinite(Number(value)) ? Number(value) : fallback;
function levelFromStorage(dam: Dam, storage: number, anchorLevel = dam.lastRecorded.level, anchorStorage = dam.currentStorage ?? storage) {
  const liveStorage = Math.max(0.001, dam.liveStorage);
  const levelRange = dam.maxLevel - dam.minLevel;
  return anchorLevel + ((storage - anchorStorage) / liveStorage) * levelRange;
}
function timeseriesLevel(row: Record<string, unknown>, dam: Dam) {
  const observed = safe(row.reservoir_level_ft);
  if (observed > 0) return observed;
  return levelFromStorage(dam, safe(row.storage_tmc, dam.currentStorage ?? 0));
}

function adaptReservoir(item: Record<string, any>): Dam {
  const capacity = safe(item.full_reservoir_level_ft, safe(item.water_level_ft, 1800));
  const level = safe(item.water_level_ft, capacity * 0.82);
  const location = item.district ?? item.sub_basin ?? item.basin ?? 'Karnataka';
  const coordinates = item.coordinates ?? {};
  return damSchema.parse({
    id: item.id, name: item.short_name ?? item.name, reservoir: item.name ?? item.short_name,
    river: item.river ?? 'River basin', location: `${location}, Karnataka`, district: location,
    state: 'Karnataka', lat: safe(coordinates.lat, 14.5), lng: safe(coordinates.lon ?? coordinates.lng, 75.8),
    yearBuilt: undefined, purpose: item.purpose ?? 'Water supply and irrigation',
    maxLevel: capacity, minLevel: capacity * 0.68, unit: 'ft',
    grossStorage: safe(item.gross_capacity_tmc), liveStorage: safe(item.live_capacity_tmc), currentStorage: item.current_storage_tmc == null ? undefined : safe(item.current_storage_tmc),
    deadStorage: safe(item.dead_storage_tmc, safe(item.live_capacity_tmc) * 0.08),
    catchmentArea: safe(item.catchment_area_sq_km), nearbyPlaces: [location.split('/')[0].trim()],
    affectedRegions: [location.split('/')[0].trim(), location.split('/').at(-1)?.trim() ?? 'Karnataka'],
    lastRecorded: { date: item.latest_date ?? new Date().toISOString().slice(0, 10), level },
  });
}

export async function fetchDams(signal?: AbortSignal): Promise<Dam[]> {
  if (USE_MOCK) return mockDams;
  try {
    const values = await readJson<unknown[]>(`${base}/api/dams`, z.array(z.unknown()), signal);
    return values.map((value) => damSchema.parse(value));
  } catch {
    const values = await readJson<Record<string, unknown>[]>(`${base}/api/reservoirs`, z.array(z.record(z.string(), z.unknown())), signal);
    return values.map(adaptReservoir);
  }
}

export async function fetchHistory(dam: Dam, asOf: string, days: number, signal?: AbortSignal): Promise<LevelPoint[]> {
  if (USE_MOCK) return makeHistory(dam, asOf, days);
  const from = new Date(`${asOf}T12:00:00`); from.setDate(from.getDate() - days + 1);
  try {
    return await readJson(`${base}/api/dams/${dam.id}/levels?from=${iso(from)}&to=${asOf}`, z.array(levelSchema), signal);
  } catch {
    try {
      const rows = await readJson(`${base}/api/reservoirs/${dam.id}?history_days=365`, z.object({ metadata: z.record(z.string(), z.unknown()), timeseries: z.array(z.record(z.string(), z.unknown())) }), signal);
      return rows.timeseries.map((row) => ({ date: String(row.date), level: timeseriesLevel(row, dam) })).filter((point) => point.date <= asOf).slice(-days);
    } catch {
      throw new Error('Could not load reservoir observations. Check the API connection and retry.');
    }
  }
}

export async function fetchForecast(dam: Dam, history: LevelPoint[], asOf: string, horizon: Horizon, signal?: AbortSignal): Promise<Forecast> {
  if (USE_MOCK) return makeForecast(dam, history, asOf, horizon);
  // A forecast is live only when its anchor is this reservoir's own latest
  // observed date. A later date from another reservoir is not a valid anchor.
  const isLatest = asOf === dam.lastRecorded.date;
  try {
      const result = await fetch(`${base}/api/predict?dam_id=${dam.id}&as_of=${asOf}&horizon=${horizon}`, { signal });
      if (result.ok) {
        const body = await result.json() as Record<string, any>;
        const predictions = (body.predictions ?? []).map((point: any) => ({ date: point.date, level: safe(point.level), lower: point.lower, upper: point.upper }));
        if (predictions.length) return forecastSchema.parse({ damId: dam.id, asOf, horizon, predictions, confidence: normalizeScores(body.confidence), persistence: normalizeValues(body.persistence, history.at(-1)?.level ?? dam.lastRecorded.level), actuals: body.actuals });
      }
  } catch { /* The current service has a separate reservoir forecast route. */ }
  let liveForecastFailure: unknown;
  if(isLatest){
    try {
      const body = await readJson(`${base}/api/reservoirs/${dam.id}/forecast`, z.record(z.string(), z.unknown()), signal);
      const daily = (body.daily_trajectory_30d as Record<string, unknown>[] | undefined) ?? [];
      const anchorPoint = history.find((point) => point.date === asOf);
      const anchorLevel = anchorPoint?.level ?? (asOf === dam.lastRecorded.date ? dam.lastRecorded.level : undefined);
      if (anchorLevel === undefined) throw new Error(`No ${dam.name} observation is available for ${asOf}.`);
      const anchorStorage = safe(body.current_storage_tmc, dam.currentStorage ?? 0);
      const predictions = daily.filter((p) => String(p.date) > asOf && Math.round(safe(p.day_offset)) <= horizon).map((p) => ({ date: String(p.date), level: levelFromStorage(dam, safe(p.predicted_storage_tmc), anchorLevel, anchorStorage), lower: undefined, upper: undefined }));
      const sourceStrategies = body.strategies as Record<string, unknown> | undefined;
      if (predictions.length) return forecastSchema.parse({ damId: dam.id, asOf, horizon, predictions, confidence: normalizeScores({}), persistence: normalizeValues({}, history.at(-1)?.level ?? dam.lastRecorded.level), strategies: sourceStrategies ? { 1: String(sourceStrategies['1'] ?? 'persistence'), 7: String(sourceStrategies['7'] ?? 'persistence'), 14: String(sourceStrategies['14'] ?? 'persistence'), 30: String(sourceStrategies['30'] ?? 'persistence') } : undefined });
      throw new Error('The backend returned no forecast points.');
    } catch (error) { liveForecastFailure = error; }
  }
  if(isLatest)throw new Error(liveForecastFailure instanceof Error?`Live forecast unavailable: ${liveForecastFailure.message}`:'Live forecast unavailable. Check that the backend model is ready.');
  const estimate=makeForecast(dam,history,asOf,horizon);
  const actuals=await fetchActuals(dam,asOf,horizon,signal);
  return {...estimate,actuals};
}

async function fetchActuals(dam:Dam,asOf:string,horizon:Horizon,signal?:AbortSignal):Promise<LevelPoint[]|undefined>{
  const fromDate=new Date(`${asOf}T12:00:00`);fromDate.setDate(fromDate.getDate()+1);const through=shiftDate(asOf,horizon);
  try{return await readJson(`${base}/api/dams/${dam.id}/levels?from=${iso(fromDate)}&to=${through}`,z.array(levelSchema),signal);}catch{
    try{const rows=await readJson(`${base}/api/reservoirs/${dam.id}?history_days=365`,z.object({metadata:z.record(z.string(),z.unknown()),timeseries:z.array(z.record(z.string(),z.unknown()))}),signal);return rows.timeseries.map((row)=>({date:String(row.date),level:timeseriesLevel(row,dam)})).filter((point)=>point.date>asOf&&point.date<=through);}catch{return undefined;}
  }
}

function normalizeScores(source: any): Record<Horizon, number> { return { 1: safe(source?.['1']), 7: safe(source?.['7']), 14: safe(source?.['14']), 30: safe(source?.['30']) }; }
function normalizeValues(source: any, current: number): Record<Horizon, number> { return { 1: safe(source?.['1'], current), 7: safe(source?.['7'], current), 14: safe(source?.['14'], current), 30: safe(source?.['30'], current) }; }
function iso(date: Date) { return date.toISOString().slice(0, 10); }
function shiftDate(value:string,days:number){const date=new Date(`${value}T12:00:00`);date.setDate(date.getDate()+days);return iso(date);}
