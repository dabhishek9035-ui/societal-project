export const GITHUB_URL = 'https://github.com/dabhishek9035-ui/societal-project';
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';
export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === 'true';

export const projectInfo = {
  name: 'Jaladrishti',
  eyebrow: 'KARNATAKA · RESERVOIR INTELLIGENCE',
  tagline: 'See the waterline before it moves.',
  highlight: 'A clearer view of what comes next. Explore daily reservoir signals, compare 1, 7, 14 and 30-day outlooks, and understand where changing water levels could matter most.',
  description: 'Jaladrishti brings reservoir observations and machine-learning forecasts into one calm, decision-ready view. Daily level histories are paired with seasonal signals and a recurrent forecasting model to help water managers explore what may happen next, understand uncertainty, and review performance against observed readings.',
  dataSummary: 'Ten major Karnataka reservoirs · daily observations · 1–30 day outlooks',
  modelSummary: 'A Delta-LSTM forecast is compared with a persistence baseline and historical observations. Forecast ranges communicate uncertainty; they are guidance for situational awareness, not an operational release instruction.',
  repository: 'dabhishek9035-ui / societal-project',
  repositoryBlurb: 'A hydrological forecasting platform for Karnataka’s ten major reservoirs, pairing daily observation histories with machine-learning outlooks.',
};
