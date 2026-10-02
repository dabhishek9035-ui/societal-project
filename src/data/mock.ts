import type { Dam, Forecast, ForecastPoint, Horizon, LevelPoint } from './types';

const rawDams = [
  ['almatti','Almatti Dam','Lal Bahadur Shastri Sagar','Krishna','Vijayapura / Bagalkote',16.331,75.888,1708,1570,123.08,2.2,123.08,36000,'Irrigation · hydropower · drinking water',['Almatti','Muddebihal'],['Bagalkote','Vijayapura','Raichur']],
  ['bhadra','Bhadra Dam','Bhadra Reservoir','Bhadra','Chikkamagaluru',13.702,75.642,2158,2070,71.5,3.8,71.5,19680,'Irrigation · hydropower',['Lakkavalli','Bhadravathi'],['Shivamogga','Chikkamagaluru']],
  ['hemavathy','Hemavathy Reservoir','Gorur Reservoir','Hemavathi','Hassan',12.909,76.053,2922,2780,37.1,1.6,37.1,28000,'Irrigation · drinking water',['Gorur','Hassan'],['Hassan','Mandya']],
  ['kabini','Kabini Reservoir','Beechanahalli Reservoir','Kabini','Mysuru',11.973,76.353,2284,2180,19.5,1.1,19.5,21400,'Irrigation · Bengaluru water supply',['H D Kote','Nanjangud'],['Mysuru','Chamarajanagar']],
  ['krsagara','Krishna Raja Sagara','KRS Reservoir','Cauvery','Mandya',12.424,76.572,124.8,102,105.79,8.4,105.79,10700,'Drinking water · irrigation',['Srirangapatna','Mysuru'],['Mandya','Mysuru']],
  ['linganamakki','Linganamakki Dam','Linganamakki Reservoir','Sharavathi','Shivamogga',14.197,74.843,1819,1680,156.61,5.8,156.61,19900,'Hydroelectric generation',['Jog Falls','Sagara'],['Shivamogga','Uttara Kannada']],
  ['malaprabha','Malaprabha Reservoir','Renuka Sagar','Malaprabha','Belagavi',15.823,75.118,2079.5,1960,37.73,2.7,37.73,11330,'Irrigation · regional water supply',['Saundatti','Naviluteertha'],['Belagavi','Bagalkote']],
  ['supa','Supa Dam','Kalinadi Reservoir','Kali','Uttara Kannada',15.275,74.526,1850,1710,147.53,6.7,147.53,10500,'Hydroelectric generation',['Joida','Dandeli'],['Uttara Kannada']],
  ['tungabhadra','Tungabhadra Dam','Pampa Sagar','Tungabhadra','Vijayanagara',15.263,76.335,1633,1510,100.8,5.2,100.8,28000,'Irrigation · hydropower',['Hosapete','Hampi'],['Vijayanagara','Koppal','Raichur']],
  ['vanivilasa_sagar','Vani Vilasa Sagara','Mari Kanive Reservoir','Vedavathi','Chitradurga',13.987,76.489,130,102,30.4,2.1,30.4,4890,'Irrigation · drought resilience',['Hiriyur','Challakere'],['Chitradurga']],
] as const;

const seed = (s: string) => [...s].reduce((n, c) => (n * 31 + c.charCodeAt(0)) >>> 0, 19);
const iso = (date: Date) => date.toISOString().slice(0, 10);
const dayOfYear = (date: Date) => Math.floor((+date - +new Date(date.getFullYear(), 0, 0)) / 86400000);

function levelAt(dam: Dam, date: Date, index: number) {
  const key = seed(dam.id);
  const capacity = dam.maxLevel - dam.minLevel;
  const season = Math.sin((dayOfYear(date) / 365.25) * Math.PI * 2 - 1.05 + (key % 11) * 0.08);
  const weekly = Math.sin(index * 0.18 + key) * capacity * 0.012;
  const slow = Math.sin(index * 0.027 + key * 0.14) * capacity * 0.035;
  const fraction = Math.max(0.38, Math.min(0.89, 0.64 + season * 0.16 + slow / capacity));
  return Math.round((dam.minLevel + capacity * fraction + weekly) * 100) / 100;
}

export const dams: Dam[] = rawDams.map(([id, short, reservoir, river, district, lat, lng, maxLevel, minLevel, grossStorage, deadStorage, liveStorage, catchmentArea, purpose, nearbyPlaces, affectedRegions], idx) => {
  const dam = { id, name: short, reservoir, river: `${river} River`, location: `${district}, Karnataka`, district, state: 'Karnataka', lat, lng, yearBuilt: [1964,1965,1979,1974,1932,1964,1972,1980,1953,1907][idx], purpose, maxLevel, minLevel, unit: 'ft', grossStorage, liveStorage, currentStorage: 0, deadStorage, catchmentArea, nearbyPlaces: [...nearbyPlaces], affectedRegions: [...affectedRegions], lastRecorded: { date: '', level: 0 } } satisfies Dam;
  const date = new Date();
  date.setHours(0, 0, 0, 0);
  dam.lastRecorded = { date: iso(date), level: levelAt(dam, date, 1400) };
  dam.currentStorage = dam.deadStorage + ((dam.lastRecorded.level-dam.minLevel)/(dam.maxLevel-dam.minLevel))*dam.liveStorage;
  return dam;
});

export function makeHistory(dam: Dam, to: string, days: number): LevelPoint[] {
  const end = new Date(`${to}T12:00:00`);
  return Array.from({ length: days }, (_, i) => {
    const date = new Date(end.getTime());
    date.setDate(date.getDate() - days + i + 1);
    const idx = Math.round((+date - +new Date(dam.lastRecorded.date)) / 86400000) + 1400;
    return { date: iso(date), level: levelAt(dam, date, idx) };
  });
}

export function makeForecast(dam: Dam, history: LevelPoint[], asOf: string, horizon: Horizon): Forecast {
  const current = history.find((p) => p.date === asOf)?.level ?? history.at(-1)?.level ?? dam.lastRecorded.level;
  const prev = history.at(-2)?.level ?? current;
  const trend = Math.max(-0.8, Math.min(0.8, (current - prev) * 0.55));
  const span = dam.maxLevel - dam.minLevel;
  const days = Array.from({ length: horizon }, (_, i) => i + 1);
  const predictions: ForecastPoint[] = days.map((day) => {
    const date = new Date(`${asOf}T12:00:00`); date.setDate(date.getDate() + day);
    const seasonal = Math.sin((dayOfYear(date) / 365.25) * Math.PI * 2 - 1.05 + seed(dam.id) % 8 * 0.1) * span * 0.0008;
    const level = Math.max(dam.minLevel, Math.min(dam.maxLevel + span * 0.025, current + trend * Math.sqrt(day) + seasonal));
    const width = span * (0.006 + day / 30 * 0.055);
    return { date: iso(date), level: Math.round(level * 100) / 100, lower: Math.round((level - width) * 100) / 100, upper: Math.round((level + width) * 100) / 100 };
  });
  const confidence: Record<Horizon, number> = { 1: 0.96, 7: 0.89, 14: 0.82, 30: 0.74 };
  const persistence: Record<Horizon, number> = { 1: current, 7: current, 14: current, 30: current };
  const actuals = predictions.map((point) => {
    const date = new Date(`${point.date}T12:00:00`);
    const index = Math.round((+date - +new Date(dam.lastRecorded.date)) / 86400000) + 1400;
    const level = levelAt(dam, date, index);
    return { date: point.date, level };
  });
  return { damId: dam.id, asOf, horizon, source: 'demo', predictions, confidence, persistence, actuals: asOf < dams[0].lastRecorded.date ? actuals : undefined };
}
