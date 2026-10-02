'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Area, Brush, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CalendarDays, ChevronLeft, ChevronRight, CircleHelp, Clock3, MapPin, Play, RotateCcw, Search, Square, Waves } from 'lucide-react';
import { fetchDams, fetchForecast, fetchHistory } from '@/data/api';
import { USE_MOCK } from '@/config';
import type { Dam, ForecastSource, Horizon, LevelPoint } from '@/data/types';
import { formatDate, Eyebrow, Meter, PageHeading, Reveal } from '@/components/Ui';
import { useAppStore } from '@/state/store';

/* ───────────────────────── helpers ───────────────────────── */

const HORIZONS: Horizon[] = [1, 7, 14, 30];

const clampDate = (value: string, min: string, max: string) => (value < min ? min : value > max ? max : value);
const shiftDate = (value: string, amount: number) => {
  const d = new Date(`${value}T12:00:00`);
  d.setDate(d.getDate() + amount);
  return d.toISOString().slice(0, 10);
};
const dateToX = (value: string) => new Date(`${value}T12:00:00`).getTime();
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const fmt = (n: number) => n.toLocaleString('en-IN');

type Metric = { date: string; predicted: number; actual: number; persistence: number; source: ForecastSource };
type ChartPoint = {
  x: number;
  date: string;
  observed: number | null;
  forecast: number | null;
  lower: number | null;
  band: number | null;
  actual: number | null;
};

/* Sine-wave path used for the water surface. The pattern repeats every `period`
   units, and the svg is 200% wide, so sliding it by -50% loops seamlessly. */
const WAVE_W = 1600;
const WAVE_H = 44;
const wavePath = (amp: number, period: number, base: number) => {
  let d = `M0 ${base}`;
  for (let x = 0; x <= WAVE_W; x += 10) {
    d += ` L${x} ${(base + Math.sin((x / period) * Math.PI * 2) * amp).toFixed(2)}`;
  }
  return `${d} L${WAVE_W} ${WAVE_H} L0 ${WAVE_H} Z`;
};
const WAVE_BACK = wavePath(7, 800, 22);
const WAVE_FRONT = wavePath(5, 533.33, 24);

/* ───────── ambient FX (deterministic so SSR and client match) ───────── */

function seeded(seed: number) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FX = (() => {
  const r = seeded(11);
  const f = (a: number, b: number) => a + r() * (b - a);
  const n = (v: number, d = 2) => v.toFixed(d);
  return {
    rays: Array.from({ length: 7 }, () => ({
      left: `${n(f(-4, 96), 1)}%`, w: `${n(f(70, 190), 0)}px`, rot: `${n(f(13, 25), 1)}deg`,
      dur: `${n(f(7, 13), 1)}s`, delay: `-${n(f(0, 9), 1)}s`, o: n(f(0.45, 1)),
    })),
    snow: Array.from({ length: 48 }, () => ({
      left: `${n(f(0, 100), 1)}%`, top: `${n(f(0, 100), 1)}%`, s: `${n(f(1.2, 3.6))}px`,
      dur: `${n(f(14, 32), 1)}s`, delay: `-${n(f(0, 30), 1)}s`, dx: `${n(f(-30, 30), 0)}px`,
    })),
    bubbles: Array.from({ length: 16 }, () => ({
      left: `${n(f(2, 98), 1)}%`, s: `${n(f(3, 10), 1)}px`, dur: `${n(f(7, 16), 1)}s`,
      delay: `-${n(f(0, 16), 1)}s`, sway: `${n(f(6, 18), 0)}px`,
    })),
    fish: [
      { top: '30%', dur: '46s', delay: '-8s', s: 1, rev: false },
      { top: '52%', dur: '62s', delay: '-31s', s: 0.7, rev: true },
      { top: '72%', dur: '54s', delay: '-20s', s: 0.55, rev: false },
      { top: '42%', dur: '78s', delay: '-50s', s: 0.85, rev: true },
    ],
  };
})();

/* Tileable caustic network, drifted in two layers in CSS. */
const CAUSTIC_SVG =
  `<svg xmlns='http://www.w3.org/2000/svg' width='360' height='360'>` +
  `<filter id='c' x='0' y='0' width='100%' height='100%' color-interpolation-filters='sRGB'>` +
  `<feTurbulence type='turbulence' baseFrequency='0.011 0.017' numOctaves='2' seed='4' stitchTiles='stitch'/>` +
  `<feColorMatrix values='0 0 0 0 .70  0 0 0 0 .97  0 0 0 0 1  -22 0 0 0 3.4'/></filter>` +
  `<rect width='100%' height='100%' filter='url(#c)'/></svg>`;
const CAUSTIC_URI = `data:image/svg+xml;utf8,${encodeURIComponent(CAUSTIC_SVG)}`;

const FISH_PATH = 'M0 8C9-2 22-2 30 8 22 18 9 18 0 8ZM30 8 42 0 39 8 42 16Z';

/* ───────────────────────── styles ───────────────────────── */

const PAGE_CSS = `
.prediction-page{
  --cy:#5de1ee;--cy2:#b4f6fa;--ink:#eefcfd;--mut:#9dbcc4;--dim:#6f8f98;
  --line:rgba(150,230,242,.18);--mono:var(--font-mono,ui-monospace,SFMono-Regular,Menlo,Consolas,monospace);
  position:relative;z-index:1;max-width:1480px;margin:0 auto;
  padding:0 clamp(16px,3.4vw,56px) 90px;color:var(--ink);
}
.prediction-page p{color:var(--mut);margin:0;line-height:1.55}
.prediction-page h2{margin:4px 0 2px;font-size:clamp(20px,1.7vw,26px);font-weight:600;letter-spacing:-.01em;color:var(--ink)}

/* ── water field: fixed behind everything, height = reservoir fill ── */
.water-field{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;
  background:
    radial-gradient(120% 80% at 50% 0%,#0a2a34 0%,#051820 55%,#030f14 100%);}
.water-body{position:absolute;left:0;right:0;bottom:0;
  transition:height 1.6s cubic-bezier(.22,.8,.24,1);
  background:linear-gradient(180deg,rgba(64,196,214,.50) 0%,rgba(16,104,126,.62) 30%,rgba(4,40,54,.88) 100%);}
.water-body::after{content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(90deg,rgba(180,246,250,.05) 0 2px,transparent 2px 120px);
  mask-image:linear-gradient(180deg,#000,transparent 70%);-webkit-mask-image:linear-gradient(180deg,#000,transparent 70%)}
.water-wave{position:absolute;left:0;bottom:calc(100% - 1px);width:200%;height:${WAVE_H}px;will-change:transform}
.water-wave.back{fill:rgba(64,196,214,.30);animation:pp-wave 16s linear infinite;bottom:calc(100% - 3px)}
.water-wave.front{fill:rgba(64,196,214,.50);animation:pp-wave 10s linear infinite reverse}
@keyframes pp-wave{to{transform:translateX(-50%)}}
.water-surface-glow{position:absolute;left:0;right:0;top:-1px;height:2px;background:linear-gradient(90deg,transparent,rgba(180,246,250,.8),transparent)}

.predicted-band,.predicted-line{position:absolute;left:0;right:0;transition:bottom 1.6s cubic-bezier(.22,.8,.24,1),height 1.6s cubic-bezier(.22,.8,.24,1)}
.predicted-band{background:repeating-linear-gradient(135deg,rgba(210,163,255,.12) 0 8px,rgba(210,163,255,.04) 8px 16px)}
.predicted-line{height:0;border-top:2px dashed rgba(210,163,255,.85)}
.predicted-line span{position:absolute;right:clamp(12px,2vw,28px);transform:translateY(-100%);padding:3px 9px;margin-top:-4px;
  font:500 11px var(--mono);letter-spacing:.06em;color:#e8d3ff;background:rgba(18,8,32,.72);border:1px solid rgba(210,163,255,.4);border-radius:8px}

.water-gauge{position:absolute;left:0;top:0;bottom:0;width:64px}
.water-gauge .tick{position:absolute;left:10px;display:flex;align-items:center;gap:6px;font:500 9px var(--mono);color:rgba(190,235,242,.45);transform:translateY(50%)}
.water-gauge .tick::before{content:'';width:10px;height:1px;background:rgba(190,235,242,.4)}
.water-gauge .marker{position:absolute;left:0;transform:translateY(50%);transition:bottom 1.6s cubic-bezier(.22,.8,.24,1);
  display:flex;align-items:center;font:600 11px var(--mono);color:#02161b}
.water-gauge .marker b{padding:4px 9px 4px 8px;background:var(--cy2);border-radius:0 8px 8px 0;white-space:nowrap}

/* ── heading ── */
.prediction-page .heading-row{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;flex-wrap:wrap;margin-bottom:26px;text-shadow:0 2px 24px rgba(2,16,22,.6)}
.prediction-page .asof-status{display:flex;align-items:center;gap:10px;padding:10px 16px;border:1px solid var(--line);border-radius:999px;
  background:rgba(3,14,19,.62);backdrop-filter:blur(8px);font:500 12px var(--mono);color:var(--mut);white-space:nowrap}
.prediction-page .asof-status strong{color:var(--cy2);font-weight:600}
.prediction-page .status-pulse{width:7px;height:7px;border-radius:50%;background:var(--cy);box-shadow:0 0 0 0 rgba(93,225,238,.6);animation:pp-pulse 2s infinite}
@keyframes pp-pulse{70%{box-shadow:0 0 0 9px rgba(93,225,238,0)}100%{box-shadow:0 0 0 0 rgba(93,225,238,0)}}

/* ── cards ── */
.prediction-page .glass-card{position:relative;background:linear-gradient(180deg,rgba(5,20,27,.78),rgba(3,11,16,.7));
  border:1px solid var(--line);border-radius:18px;padding:20px 22px;
  backdrop-filter:blur(12px) saturate(1.2);-webkit-backdrop-filter:blur(12px) saturate(1.2);
  box-shadow:0 22px 50px -28px rgba(0,0,0,.95),inset 0 1px 0 rgba(160,240,250,.07)}
.prediction-page .card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.prediction-page .data-label{display:block;font:500 10px var(--mono);letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.prediction-page .tag{font:500 10px var(--mono);letter-spacing:.12em;color:var(--cy);padding:6px 11px;border:1px solid rgba(93,225,238,.28);border-radius:999px;background:rgba(93,225,238,.06);white-space:nowrap}

/* ── control deck ── */
.prediction-page .deck-wrap{position:relative;z-index:30;margin-bottom:18px}
.prediction-page .control-deck{display:grid;grid-template-columns:minmax(260px,1.3fr) minmax(280px,1fr) minmax(220px,.8fr);gap:22px;align-items:end;padding:18px 22px}
.prediction-page .control{display:flex;flex-direction:column;gap:9px;min-width:0}
.prediction-page .control-label{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:13px;color:var(--mut)}
.prediction-page .control-label strong{font:600 13px var(--mono);color:var(--cy2)}

.prediction-page .dam-picker{position:relative;display:flex;align-items:center;gap:10px;padding:0 6px 0 14px;height:50px;border-radius:13px;background:rgba(2,10,14,.6);border:1px solid var(--line);color:var(--mut);transition:.2s}
.prediction-page .dam-picker:focus-within{border-color:rgba(93,225,238,.6);box-shadow:0 0 0 4px rgba(93,225,238,.09)}
.prediction-page .dam-picker input{flex:1;min-width:0;background:none;border:0;outline:0;color:var(--ink);font-size:16px;font-weight:500}
.prediction-page .dam-picker>button{width:38px;height:38px;border-radius:10px;border:0;background:transparent;color:var(--cy);font-size:18px;cursor:pointer}
.prediction-page .dam-picker>button:hover{background:rgba(93,225,238,.1)}
.prediction-page .dam-options{position:absolute;left:0;right:0;top:calc(100% + 8px);max-height:340px;overflow:auto;padding:6px;border-radius:16px;background:rgba(3,12,17,.97);border:1px solid rgba(125,222,238,.28);box-shadow:0 30px 60px -20px #000;z-index:50;scrollbar-width:thin}
.prediction-page .dam-options button{width:100%;display:flex;align-items:center;gap:12px;text-align:left;padding:10px 12px;border:0;border-radius:11px;background:transparent;color:var(--ink);cursor:pointer}
.prediction-page .dam-options button:hover,.prediction-page .dam-options button.selected{background:rgba(93,225,238,.1)}
.prediction-page .dam-options button span:nth-child(2){flex:1;display:flex;flex-direction:column;min-width:0}
.prediction-page .dam-options small{color:var(--dim);font-size:11.5px}
.prediction-page .option-monogram{width:32px;height:32px;display:grid;place-items:center;border-radius:9px;background:rgba(93,225,238,.12);color:var(--cy);font-weight:600}
.prediction-page .option-value{font:500 11px var(--mono);color:var(--mut);white-space:nowrap}
.prediction-page .options-empty{padding:16px;color:var(--mut);font-size:13px}

.prediction-page .segmented-control{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:5px;height:50px;border-radius:13px;background:rgba(2,10,14,.55);border:1px solid var(--line)}
.prediction-page .segmented-control button{display:flex;align-items:baseline;justify-content:center;gap:5px;border:0;border-radius:9px;background:transparent;color:var(--mut);cursor:pointer;transition:.2s}
.prediction-page .segmented-control button strong{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums;align-self:center}
.prediction-page .segmented-control button span{font-size:11px;align-self:center}
.prediction-page .segmented-control button:hover{background:rgba(93,225,238,.08);color:var(--ink)}
.prediction-page .segmented-control button.selected{background:linear-gradient(180deg,#a4f4f8,#5de1ee);color:#021419;box-shadow:0 8px 26px -10px rgba(93,225,238,.7)}

.prediction-page .range-slider{-webkit-appearance:none;appearance:none;width:100%;height:6px;margin:22px 0 14px;border-radius:99px;background:linear-gradient(90deg,rgba(93,225,238,.5),rgba(93,225,238,.12));outline:0;cursor:pointer}
.prediction-page .range-slider::-webkit-slider-thumb{-webkit-appearance:none;width:20px;height:20px;border-radius:50%;background:var(--cy);border:3px solid #021419;box-shadow:0 0 0 1px var(--cy),0 0 16px rgba(93,225,238,.7)}
.prediction-page .range-slider::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--cy);border:3px solid #021419;box-shadow:0 0 0 1px var(--cy)}

/* ── KPI strip ── */
.prediction-page .kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
.prediction-page .kpi{display:flex;flex-direction:column;gap:5px;padding:16px 18px}
.prediction-page .kpi strong{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
.prediction-page .kpi strong small{font-size:13px;color:var(--mut);font-weight:400}
.prediction-page .kpi>span:last-child{font-size:12px;color:var(--mut)}
.prediction-page .kpi.accent{border-color:rgba(93,225,238,.4);background:linear-gradient(135deg,rgba(93,225,238,.16),rgba(3,11,16,.72))}
.prediction-page .kpi .up{color:#ff9aa1}.prediction-page .kpi .down{color:#6df0b0}
.prediction-page .kpi-toggle{justify-content:space-between}
.prediction-page .tank-toggle{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--ink);cursor:pointer;position:relative;user-select:none}
.prediction-page .tank-toggle input{position:absolute;opacity:0;width:0;height:0}
.prediction-page .switch{width:40px;height:23px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid var(--line);position:relative;transition:.25s;flex:none}
.prediction-page .switch::after{content:'';position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;background:var(--mut);transition:.25s}
.prediction-page .tank-toggle input:checked + .switch{background:rgba(210,163,255,.25);border-color:rgba(210,163,255,.6)}
.prediction-page .tank-toggle input:checked + .switch::after{transform:translateX(17px);background:#d2a3ff;box-shadow:0 0 10px #d2a3ff}
.prediction-page .tank-toggle input:focus-visible + .switch{outline:2px solid var(--cy);outline-offset:2px}

/* ── main grid ── */
.prediction-page .main-grid{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:20px;align-items:start;margin-bottom:20px}
.prediction-page .details-column{position:sticky;top:112px;max-height:calc(100vh - 132px);overflow:auto;scrollbar-width:thin;scrollbar-color:rgba(93,225,238,.3) transparent;border-radius:18px}
.prediction-page .lower-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:20px;align-items:start;margin-bottom:16px}

/* ── chart ── */
.prediction-page .chart-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--mut)}
.prediction-page .chart-legend span{display:flex;align-items:center;gap:8px}
.prediction-page .chart-legend i{display:block;width:22px;height:0;border-top:2px solid}
.prediction-page .legend-observed{border-color:#7ca4aa}
.prediction-page .legend-forecast{border-color:#5de1ee;border-top-style:dashed}
.prediction-page .legend-actual{border-color:#d2a3ff}
.prediction-page .chart-wrap{height:clamp(320px,46vh,460px);position:relative}
.prediction-page .chart-state{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--mut);text-align:center}
.prediction-page .chart-skeleton{height:100%;display:flex;flex-direction:column;justify-content:center;gap:18px}
.prediction-page .chart-skeleton span{height:14px;border-radius:8px;background:linear-gradient(90deg,rgba(93,225,238,.05),rgba(93,225,238,.18),rgba(93,225,238,.05));background-size:200% 100%;animation:pp-shimmer 1.4s infinite linear}
.prediction-page .chart-skeleton span:nth-child(2){width:78%}.prediction-page .chart-skeleton span:nth-child(3){width:55%}
@keyframes pp-shimmer{to{background-position:-200% 0}}
.prediction-page .chart-foot{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:12px;color:var(--dim)}
.prediction-page .chart-foot>span{display:flex;align-items:center;gap:7px}
.prediction-page .risk-dot{width:8px;height:8px;border-radius:2px;background:rgba(255,86,96,.55)}
.prediction-page .chart-tooltip{padding:11px 14px;border-radius:12px;background:rgba(3,12,17,.95);border:1px solid rgba(125,222,238,.3);box-shadow:0 14px 34px -10px #000;display:flex;flex-direction:column;gap:6px;min-width:190px}
.prediction-page .chart-tooltip>span{font:500 10px var(--mono);letter-spacing:.12em;color:var(--mut);text-transform:uppercase}
.prediction-page .chart-tooltip div{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--mut)}
.prediction-page .chart-tooltip div strong{margin-left:auto;color:var(--ink);font-variant-numeric:tabular-nums}
.prediction-page .chart-tooltip i{width:8px;height:8px;border-radius:50%}
.prediction-page .tooltip-observed{background:#7ca4aa}.prediction-page .tooltip-forecast{background:#5de1ee}.prediction-page .tooltip-actual{background:#d2a3ff}

/* ── model comparison ── */
.prediction-page .horizon-comparison{display:flex;flex-direction:column;gap:8px}
.prediction-page .horizon-row{display:grid;grid-template-columns:56px minmax(90px,1fr) minmax(130px,1.1fr) auto;align-items:center;gap:16px;padding:12px 14px;border-radius:14px;border:1px solid transparent;background:rgba(2,10,14,.45);color:var(--ink);text-align:left;cursor:pointer;transition:.2s}
.prediction-page .horizon-row:hover{border-color:var(--line)}
.prediction-page .horizon-row.active{border-color:rgba(93,225,238,.5);background:rgba(93,225,238,.07);box-shadow:inset 3px 0 0 var(--cy)}
.prediction-page .horizon-name{font-size:24px;font-weight:600;line-height:1;display:flex;flex-direction:column;gap:3px}
.prediction-page .horizon-name small{font:500 9px var(--mono);letter-spacing:.14em;color:var(--dim)}
.prediction-page .confidence-meter{display:flex;flex-direction:column;gap:6px}
.prediction-page .confidence-meter strong{font-size:14px}
.prediction-page .value-pair{display:flex;flex-direction:column;gap:3px;font-size:12.5px;color:var(--mut)}
.prediction-page .value-pair strong{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}
.prediction-page .skill-note{font:500 11px var(--mono);color:var(--dim);text-align:right;white-space:nowrap}
.prediction-page .skill-note.positive{color:#6df0b0}

/* ── backtest ── */
.prediction-page .reset-session{display:inline-flex;align-items:center;gap:7px;padding:8px 13px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--mut);font-size:12px;cursor:pointer;transition:.2s}
.prediction-page .reset-session:hover{color:var(--ink);border-color:rgba(93,225,238,.5)}
.prediction-page .time-controls{display:flex;align-items:stretch;gap:10px;flex-wrap:wrap}
.prediction-page .step-button,.prediction-page .latest-button,.prediction-page .play-button{display:inline-flex;align-items:center;gap:7px;height:44px;padding:0 15px;border-radius:12px;border:1px solid var(--line);background:rgba(2,10,14,.55);color:var(--ink);font-weight:500;font-size:13.5px;cursor:pointer;transition:.2s}
.prediction-page .step-button:hover:not(:disabled),.prediction-page .latest-button:hover{border-color:rgba(93,225,238,.55);background:rgba(93,225,238,.09)}
.prediction-page .step-button:disabled,.prediction-page .play-button:disabled{opacity:.35;cursor:not-allowed}
.prediction-page .play-button{background:linear-gradient(180deg,#a4f4f8,#5de1ee);border-color:transparent;color:#021419;font-weight:600;box-shadow:0 8px 24px -10px rgba(93,225,238,.8)}
.prediction-page .play-button.playing{background:rgba(255,120,130,.14);color:#ffb3b8;border:1px solid rgba(255,120,130,.4);box-shadow:none}
.prediction-page .date-picker{display:flex;align-items:center;gap:9px;height:44px;padding:0 14px;border-radius:12px;border:1px solid var(--line);background:rgba(2,10,14,.55);color:var(--cy);flex:1;min-width:170px}
.prediction-page .date-picker input{background:none;border:0;outline:0;color:var(--ink);font:500 14px var(--mono);color-scheme:dark;flex:1;min-width:0}
.prediction-page .play-speed{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0;padding:10px 14px;border-radius:12px;background:rgba(2,10,14,.4);font-size:12px;color:var(--dim)}
.prediction-page .play-speed input{flex:1;min-width:110px;accent-color:var(--cy)}
.prediction-page .play-speed strong{color:var(--cy2);font-weight:600;font-family:var(--mono)}
.prediction-page .keyboard-note{margin-left:auto}
.prediction-page .result-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.prediction-page .result-grid>div{display:flex;flex-direction:column;gap:5px;padding:13px 15px;border-radius:14px;background:rgba(2,10,14,.5);border:1px solid var(--line)}
.prediction-page .result-grid .result-current{grid-column:1/-1;background:linear-gradient(135deg,rgba(93,225,238,.1),rgba(2,10,14,.5));border-color:rgba(93,225,238,.3)}
.prediction-page .result-current strong{font-size:24px;font-weight:600}
.prediction-page .result-current>span:last-child{font-size:12px;color:var(--mut)}
.prediction-page .result-stat strong{font-size:17px}
.prediction-page .session-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px;padding:13px 15px;border-radius:14px;border:1px dashed rgba(125,222,238,.22)}
.prediction-page .session-summary>div>span{display:block;font:500 10px var(--mono);letter-spacing:.12em;color:var(--dim);margin-bottom:5px}
.prediction-page .session-summary strong{font-size:16px;font-variant-numeric:tabular-nums}
.prediction-page .session-title{grid-column:1/-1}
.prediction-page .session-title svg{width:100%;height:38px;margin-top:6px;overflow:visible}
.prediction-page .session-title b{color:var(--cy2)}
.prediction-page .footnote{display:flex;gap:14px;align-items:flex-start;padding:4px 6px;font-size:12px;text-shadow:0 1px 12px rgba(2,16,22,.9)}
.prediction-page .footnote>span{font:500 10px var(--mono);letter-spacing:.12em;color:var(--cy);padding-top:2px;white-space:nowrap}

/* ── dam details ── */
.prediction-page .details-card{display:flex;flex-direction:column;gap:16px}
.prediction-page .details-title{display:flex;justify-content:space-between;align-items:center}
.prediction-page .details-emblem{width:36px;height:36px;display:grid;place-items:center;border-radius:11px;background:rgba(93,225,238,.1);color:var(--cy);border:1px solid rgba(93,225,238,.25)}
.prediction-page .details-card h2{font-size:28px;margin:0;line-height:1.05}
.prediction-page .details-location{display:flex;align-items:center;gap:6px;font-size:13px;margin-top:-8px}
.prediction-page .details-asof{padding:14px 16px;border-radius:14px;background:linear-gradient(135deg,rgba(93,225,238,.12),rgba(93,225,238,.03));border:1px solid rgba(93,225,238,.25)}
.prediction-page .details-asof span{display:block;font:500 10px var(--mono);letter-spacing:.12em;color:var(--mut);margin-bottom:4px}
.prediction-page .details-asof strong{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}
.prediction-page .details-asof small{font-size:13px;color:var(--mut);font-weight:400}
.prediction-page .detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;border-radius:14px;overflow:hidden;background:var(--line);border:1px solid var(--line)}
.prediction-page .detail-grid>div{display:flex;flex-direction:column;gap:3px;padding:11px 13px;background:rgba(3,12,17,.9)}
.prediction-page .detail-grid span{font:500 9px var(--mono);letter-spacing:.1em;color:var(--dim)}
.prediction-page .detail-grid strong{font-size:13px;font-weight:500;line-height:1.35}
.prediction-page .places-section{display:flex;flex-direction:column;gap:9px}
.prediction-page .chip-row{display:flex;flex-wrap:wrap;gap:6px}
.prediction-page .chip{padding:5px 11px;border-radius:999px;font-size:12px;color:var(--cy2);background:rgba(93,225,238,.08);border:1px solid rgba(93,225,238,.22)}
.prediction-page .chip-alert{color:#ffb3b3;background:rgba(255,86,96,.09);border-color:rgba(255,120,130,.3)}
.prediction-page .location-map{position:relative;height:200px;border-radius:16px;overflow:hidden;border:1px solid var(--line);background:radial-gradient(circle at 50% 45%,rgba(93,225,238,.1),rgba(2,10,14,.9) 70%)}
.prediction-page .location-map::before{content:'';position:absolute;inset:0;background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:28px 28px;opacity:.45}
.prediction-page .location-map svg{position:absolute;inset:0;width:100%;height:100%}
.prediction-page .map-watermark{position:absolute;left:14px;top:10px;font:600 22px var(--mono);letter-spacing:.3em;color:rgba(93,225,238,.07)}
.prediction-page .map-outline{fill:rgba(93,225,238,.06);stroke:rgba(93,225,238,.45);stroke-width:.6}
.prediction-page .map-river{fill:none;stroke:rgba(93,225,238,.55);stroke-width:.7;stroke-dasharray:2 1.5}
.prediction-page .map-pin{fill:#a4f4f8}
.prediction-page .map-ping{fill:rgba(93,225,238,.25);animation:pp-ping 2.2s infinite;transform-box:fill-box;transform-origin:center}
@keyframes pp-ping{0%{transform:scale(.5);opacity:1}100%{transform:scale(2.4);opacity:0}}
.prediction-page .map-label{position:absolute;left:12px;bottom:10px;display:flex;flex-direction:column;gap:2px}
.prediction-page .map-label span{font-size:12px;color:var(--mut)}
.prediction-page .map-label strong{font:500 11px var(--mono);color:var(--cy2)}
.prediction-page .map-label i{display:inline-block;width:4px}
.prediction-page .details-footer{display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid var(--line);font-size:12px;color:var(--dim)}
.prediction-page .details-footer strong{color:var(--ink);font-weight:500}

/* ── ambient water FX (decorative only; never touches level geometry) ── */
.fx-layer{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.fx-sun{position:absolute;top:-18vmax;right:2%;width:46vmax;height:46vmax;border-radius:50%;
  background:radial-gradient(circle,rgba(180,246,250,.34) 0%,rgba(93,225,238,.14) 28%,transparent 62%);
  mix-blend-mode:screen;animation:pp-sun 9s ease-in-out infinite alternate}
@keyframes pp-sun{from{opacity:.65;transform:scale(.96)}to{opacity:1;transform:scale(1.04)}}

.fx-rays{mix-blend-mode:screen;
  -webkit-mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.6) 55%,transparent 95%);
  mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.6) 55%,transparent 95%)}
.fx-ray{position:absolute;top:-25%;height:150%;transform-origin:50% 0;filter:blur(9px);will-change:transform,opacity;
  background:linear-gradient(180deg,rgba(170,244,250,.26),rgba(93,225,238,.09) 50%,transparent 90%);
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 50%,transparent);
  mask-image:linear-gradient(90deg,transparent,#000 50%,transparent);
  animation:pp-ray var(--d) ease-in-out var(--dl) infinite alternate}
@keyframes pp-ray{
  from{opacity:calc(var(--o)*.35);transform:rotate(calc(var(--r) - 2.2deg))}
  to{opacity:var(--o);transform:rotate(calc(var(--r) + 2.2deg))}}

.fx-snow span{position:absolute;border-radius:50%;background:rgba(190,248,252,.8);box-shadow:0 0 6px rgba(120,235,245,.5);
  width:var(--s);height:var(--s);left:var(--l);top:var(--t);will-change:transform,opacity;
  animation:pp-snow var(--d) linear var(--dl) infinite}
@keyframes pp-snow{
  0%{transform:translate3d(0,0,0);opacity:0}
  15%{opacity:.8} 50%{opacity:.35} 85%{opacity:.75}
  100%{transform:translate3d(var(--dx),130px,0);opacity:0}}

.fx-vignette{position:absolute;inset:0;
  background:radial-gradient(ellipse 85% 75% at 50% 42%,transparent 50%,rgba(1,7,10,.6) 100%)}

/* inside the water body: clipped to the current fill, so they follow the level */
.water-fx{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.fx-caustics{position:absolute;inset:0;mix-blend-mode:screen;background-image:url("${CAUSTIC_URI}");
  -webkit-mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.55) 45%,rgba(0,0,0,.12) 100%);
  mask-image:linear-gradient(180deg,#000 0%,rgba(0,0,0,.55) 45%,rgba(0,0,0,.12) 100%)}
.fx-caustics.a{opacity:.34;background-size:360px 360px;animation:pp-caustic 42s linear infinite}
.fx-caustics.b{opacity:.22;background-size:230px 230px;animation:pp-caustic 30s linear infinite reverse}
@keyframes pp-caustic{to{background-position:360px 360px}}

.fx-bubble{position:absolute;top:104%;width:var(--s);height:var(--s);left:var(--l);opacity:0;
  animation:pp-rise var(--d) linear var(--dl) infinite}
.fx-bubble i{display:block;width:100%;height:100%;border-radius:50%;
  border:1px solid rgba(190,246,252,.55);
  background:radial-gradient(circle at 30% 30%,rgba(255,255,255,.5),rgba(150,235,245,.08) 60%);
  animation:pp-sway 3.2s ease-in-out infinite alternate}
@keyframes pp-rise{0%{top:104%;opacity:0}8%{opacity:.9}85%{opacity:.7}100%{top:-2%;opacity:0}}
@keyframes pp-sway{from{transform:translateX(calc(var(--sw)*-1))}to{transform:translateX(var(--sw))}}

.fx-fish{position:absolute;left:0;width:42px;height:16px;opacity:.5;color:rgba(130,228,240,.75);
  animation:pp-swim var(--d) linear var(--dl) infinite;will-change:transform}
.fx-fish.rev{left:auto;right:0;animation-name:pp-swim-rev}
.fx-fish svg{display:block;width:100%;height:100%;fill:currentColor;transform:scale(var(--fs));animation:pp-bob 4s ease-in-out infinite alternate}
.fx-fish.rev svg{transform:scale(calc(var(--fs)*-1),var(--fs))}
@keyframes pp-swim{from{transform:translateX(-80px)}to{transform:translateX(calc(100vw + 80px))}}
@keyframes pp-swim-rev{from{transform:translateX(80px)}to{transform:translateX(calc(-100vw - 80px))}}
@keyframes pp-bob{from{translate:0 -5px}to{translate:0 6px}}

/* brighter, travelling highlight on the surface line */
.water-surface-glow::after{content:'';position:absolute;top:-1px;height:4px;width:26%;left:-26%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.9),transparent);filter:blur(1.5px);
  animation:pp-glint 7s ease-in-out infinite}
@keyframes pp-glint{to{left:100%}}

/* ── responsive ── */
@media (max-width:1280px){
  .prediction-page .lower-grid{grid-template-columns:1fr}
}
@media (max-width:1180px){
  .prediction-page .main-grid{grid-template-columns:1fr}
  .prediction-page .details-column{position:static;max-height:none}
  .prediction-page .control-deck{grid-template-columns:1fr 1fr}
  .prediction-page .control-deck .control:first-child{grid-column:1/-1}
  .prediction-page .kpi-strip{grid-template-columns:1fr 1fr}
}
@media (max-width:760px){
  .water-gauge{display:none}
  .fx-snow span:nth-child(n+25),.fx-fish,.fx-ray:nth-child(n+5){display:none}
  .prediction-page .glass-card{padding:16px;border-radius:16px}
  .prediction-page .control-deck{grid-template-columns:1fr}
  .prediction-page .kpi-strip{grid-template-columns:1fr 1fr;gap:10px}
  .prediction-page .kpi strong{font-size:21px}
  .prediction-page .horizon-row{grid-template-columns:48px 1fr;gap:10px}
  .prediction-page .horizon-row .skill-note{text-align:left}
  .prediction-page .detail-grid{grid-template-columns:1fr}
  .prediction-page .keyboard-note{display:none}
}
@media (prefers-reduced-motion:reduce){
  .water-wave,.status-pulse,.map-ping,.fx-sun,.fx-ray,.fx-snow span,.fx-caustics,
  .fx-bubble,.fx-bubble i,.fx-fish,.fx-fish svg,.water-surface-glow::after{animation:none !important}
  .fx-bubble,.fx-fish{display:none}
  .water-body,.predicted-line,.predicted-band,.water-gauge .marker{transition:none !important}
}
`;

/* ───────────────────────── page ───────────────────────── */

export default function PredictionsPage() {
  const queryClient = useQueryClient();

  const [damId, setDamId] = useState('almatti');
  const [horizon, setHorizon] = useState<Horizon>(7);
  const [windowDays, setWindowDays] = useState(180);
  const [asOf, setAsOf] = useState('');
  const [debouncedAsOf, setDebouncedAsOf] = useState('');
  const [search, setSearch] = useState('');
  const [showDamOptions, setShowDamOptions] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(800);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [showPredicted, setShowPredicted] = useState(true);
  const [urlReady, setUrlReady] = useState(false);

  const pendingStep = useRef(false);
  const recordedSteps = useRef(new Set<string>());
  const selectorRef = useRef<HTMLDivElement>(null);

  const setSelectedDam = useAppStore((s) => s.setSelectedDam);
  const setStoreHorizon = useAppStore((s) => s.setHorizon);

  /* ── data ── */
  const damsQuery = useQuery({ queryKey: ['dams'], queryFn: ({ signal }) => fetchDams(signal) });
  const dams = damsQuery.data ?? [];
  const dam = dams.find((item) => item.id === damId) ?? dams[0];

  const latestDate = useMemo(
    () => dams.reduce((max, item) => (item.lastRecorded.date > max ? item.lastRecorded.date : max), '') || new Date().toISOString().slice(0, 10),
    [dams],
  );
  // Each reservoir can have a different latest observation date. Using the
  // global maximum makes stale reservoirs look current and can anchor a live
  // forecast to a date for which that dam has no observation.
  const damLatestDate = dam?.lastRecorded.date ?? latestDate;
  const damMinDate = shiftDate(damLatestDate, USE_MOCK ? -1459 : -364);

  /* ── URL <-> state ── */
  useEffect(() => {
    if (!dams.length || urlReady) return;
    const params = new URLSearchParams(window.location.search);
    const requestedDam = params.get('dam');
    const requestedH = Number(params.get('horizon'));
    const requestedWindow = Number(params.get('window'));

    const nextDam = dams.some((item) => item.id === requestedDam) ? requestedDam! : dams[0].id;
    const nextH = HORIZONS.includes(requestedH as Horizon) ? (requestedH as Horizon) : 7;
    const nextWindow = Number.isFinite(requestedWindow) && requestedWindow > 0 ? Math.max(120, Math.min(365, requestedWindow)) : 180;
    const initialDam = dams.find((item) => item.id === nextDam) ?? dams[0];
    const initialLatest = initialDam.lastRecorded.date;
    const initialMin = shiftDate(initialLatest, USE_MOCK ? -1459 : -364);
    const requestedDate = params.get('asOf') ?? initialLatest;
    const nextDate = clampDate(requestedDate, initialMin, initialLatest);

    setDamId(nextDam);
    setHorizon(nextH);
    setWindowDays(nextWindow);
    setAsOf(nextDate);
    setDebouncedAsOf(nextDate);
    setSelectedDam(nextDam);
    setStoreHorizon(nextH);
    setUrlReady(true);
  }, [dams, urlReady, setSelectedDam, setStoreHorizon]);

  useEffect(() => {
    if (!urlReady || !asOf) return;
    const params = new URLSearchParams();
    params.set('dam', damId);
    params.set('horizon', String(horizon));
    params.set('window', String(windowDays));
    params.set('asOf', asOf);
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
  }, [damId, horizon, windowDays, asOf, urlReady]);

  useEffect(() => {
    const id = window.setTimeout(() => setDebouncedAsOf(asOf), 160);
    return () => window.clearTimeout(id);
  }, [asOf]);

  useEffect(() => { setSelectedDam(damId); }, [damId, setSelectedDam]);
  useEffect(() => { setStoreHorizon(horizon); }, [horizon, setStoreHorizon]);
  useEffect(() => { setMetrics([]); }, [damId]);

  /* ── history + forecasts ── */
  const historyQuery = useQuery({
    queryKey: ['history', dam?.id, debouncedAsOf, windowDays],
    queryFn: ({ signal }) => fetchHistory(dam!, debouncedAsOf, windowDays, signal),
    enabled: !!dam && urlReady && !!debouncedAsOf,
  });
  const history = historyQuery.data ?? [];

  const historyThroughAsOf = history.filter((point) => point.date <= debouncedAsOf);
  const hasAnchorObservation = historyThroughAsOf.length > 0 &&
    (debouncedAsOf < (dam?.lastRecorded.date ?? '') || historyThroughAsOf.some((point) => point.date === debouncedAsOf));
  const forecastsEnabled = urlReady && historyQuery.isSuccess && hasAnchorObservation;
  const q1 = useForecastQuery(dam, history, debouncedAsOf, 1, forecastsEnabled);
  const q7 = useForecastQuery(dam, history, debouncedAsOf, 7, forecastsEnabled);
  const q14 = useForecastQuery(dam, history, debouncedAsOf, 14, forecastsEnabled);
  const q30 = useForecastQuery(dam, history, debouncedAsOf, 30, forecastsEnabled);
  const horizonQueries = [q1, q7, q14, q30];
  const forecastQuery = horizonQueries[HORIZONS.indexOf(horizon)];
  const cachedForecast = forecastQuery.data;
  // FIX: guard for missing data first. `cachedForecast?.damId === dam?.id` is
  // `undefined === undefined` (true) while both are still loading, which let the
  // `&&` chain continue into `cachedForecast.asOf` and throw.
  const forecast =
    cachedForecast &&
    dam &&
    cachedForecast.damId === dam.id &&
    cachedForecast.asOf === debouncedAsOf &&
    cachedForecast.horizon === horizon
      ? cachedForecast
      : undefined;
  const isHistoricalEstimate = forecast?.source === 'historical_estimate';
  const isDemoForecast = forecast?.source === 'demo';
  const hasForecastBand = forecast?.predictions.some((point) => point.lower !== undefined && point.upper !== undefined) ?? false;
  const forecastMethodLabel = isHistoricalEstimate
    ? 'HISTORICAL ESTIMATE'
    : isDemoForecast
      ? 'DEMO FORECAST'
      : forecast?.strategies?.[horizon] === 'recent_trend'
        ? 'TREND BASELINE'
        : forecast?.strategies?.[horizon] === 'persistence'
          ? 'PERSISTENCE'
          : 'DELTA-LSTM';

  const currentLevel = historyThroughAsOf.at(-1)?.level ?? dam?.lastRecorded.level ?? 0;
  const predictedPoint = forecast?.predictions.at(-1);
  const predictedLevel = predictedPoint?.level ?? currentLevel;

  const toFraction = (level: number) => (dam ? clamp01((level - dam.minLevel) / (dam.maxLevel - dam.minLevel)) : 0.5);
  const currentFraction = toFraction(currentLevel);
  const predictedFraction = toFraction(predictedLevel);
  const delta = predictedLevel - currentLevel;

  /* ── backtest recording ── */
  useEffect(() => {
    if (!dam || !debouncedAsOf || !forecastQuery.isSuccess || forecastQuery.isFetching || !pendingStep.current || !forecast) return;
    const key = `${dam.id}:${debouncedAsOf}:${horizon}`;
    if (recordedSteps.current.has(key)) { pendingStep.current = false; return; }
    const actualPoint = forecast.actuals?.at(-1);
    const pred = forecast.predictions.at(-1);
    if (actualPoint && pred) {
      recordedSteps.current.add(key);
      setMetrics((items) => [...items, { date: debouncedAsOf, predicted: pred.level, actual: actualPoint.level, persistence: forecast.persistence[horizon], source: forecast.source ?? 'trained_model' }].slice(-32));
    }
    pendingStep.current = false;
  }, [dam?.id, debouncedAsOf, horizon, forecastQuery.isSuccess, forecastQuery.isFetching, forecast]);

  /* ── stepping / playback / keyboard ── */
  const step = useCallback((amount: number) => {
    setAsOf((date) => {
      if (!date) return date;
      const capped = clampDate(shiftDate(date, amount), damMinDate, damLatestDate);
      if (capped === date) return date;
      pendingStep.current = true;
      return capped;
    });
  }, [damMinDate, damLatestDate]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setAsOf((date) => {
        if (!date || date >= damLatestDate) { setPlaying(false); return date; }
        pendingStep.current = true;
        return shiftDate(date, 1);
      });
    }, playSpeed);
    return () => window.clearInterval(timer);
  }, [playing, damLatestDate, playSpeed]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const t = event.target;
      if (t instanceof HTMLElement && (['INPUT', 'SELECT', 'TEXTAREA'].includes(t.tagName) || t.isContentEditable)) return;
      if (event.key === 'ArrowLeft') { event.preventDefault(); step(-1); }
      if (event.key === 'ArrowRight') { event.preventDefault(); step(1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step]);

  useEffect(() => {
    const onOutside = (event: MouseEvent) => {
      if (selectorRef.current && !selectorRef.current.contains(event.target as Node)) setShowDamOptions(false);
    };
    document.addEventListener('mousedown', onOutside);
    return () => document.removeEventListener('mousedown', onOutside);
  }, []);

  /* prefetch the next day so playback feels instant */
  useEffect(() => {
    if (!dam || !debouncedAsOf || !forecastQuery.isSuccess) return;
    const next = shiftDate(debouncedAsOf, 1);
    if (next > damLatestDate) return;
    queryClient
      .fetchQuery({ queryKey: ['history', dam.id, next, windowDays], queryFn: ({ signal }) => fetchHistory(dam, next, windowDays, signal) })
      .then((nextHistory) =>
        queryClient.prefetchQuery({ queryKey: ['forecast', dam.id, next, horizon], queryFn: ({ signal }) => fetchForecast(dam, nextHistory, next, horizon, signal) }),
      );
  }, [dam?.id, debouncedAsOf, horizon, windowDays, forecastQuery.isSuccess, damLatestDate, queryClient]);

  /* ── chart data ── */
  const chartData = useMemo(() => {
    const rows: ChartPoint[] = history
      .filter((p) => p.date <= debouncedAsOf)
      .map((p) => ({ x: dateToX(p.date), date: p.date, observed: p.level, forecast: null, lower: null, band: null, actual: null }));
    const anchor = rows.at(-1);
    if (anchor) anchor.forecast = anchor.observed;
    const actualByDate = new Map(forecast?.actuals?.map((p) => [p.date, p.level]) ?? []);
    forecast?.predictions.forEach((p) =>
      rows.push({
        x: dateToX(p.date),
        date: p.date,
        observed: null,
        forecast: p.level,
        lower: p.lower ?? null,
        band: p.lower !== undefined && p.upper !== undefined ? p.upper - p.lower : null,
        actual: actualByDate.get(p.date) ?? null,
      }),
    );
    return rows;
  }, [history, debouncedAsOf, forecast]);

  const chartYDomain = useMemo<[number, number]>(() => {
    if (!dam) return [0, 1];
    const values = chartData
      .flatMap((p) => [p.observed, p.forecast, p.actual, p.lower, p.lower !== null && p.band !== null ? p.lower + p.band : null])
      .filter((v): v is number => v !== null && Number.isFinite(v));
    if (!values.length) return [dam.minLevel, dam.maxLevel];
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const span = Math.max(dataMax - dataMin, Math.max(1, dam.maxLevel - dam.minLevel) * 0.025);
    const padding = span * 0.14;
    return [dataMin - padding, dataMax + padding];
  }, [chartData, dam]);

  const yDecimals = useMemo(() => {
    const span = chartYDomain[1] - chartYDomain[0];
    return span < 1 ? 3 : span < 10 ? 2 : span < 100 ? 1 : 0;
  }, [chartYDomain]);

  /* ── backtest stats ── */
  const actual = forecast?.actuals?.at(-1);
  const modelError = actual && predictedPoint ? Math.abs(predictedPoint.level - actual.level) : undefined;
  const persistence = forecast?.persistence[horizon];
  const persistenceError = actual && persistence !== undefined ? Math.abs(persistence - actual.level) : undefined;
  const mae = metrics.length ? metrics.reduce((s, m) => s + Math.abs(m.predicted - m.actual), 0) / metrics.length : 0;
  const rmse = metrics.length ? Math.sqrt(metrics.reduce((s, m) => s + (m.predicted - m.actual) ** 2, 0) / metrics.length) : 0;
  const beats = metrics.length ? (metrics.filter((m) => Math.abs(m.predicted - m.actual) < Math.abs(m.persistence - m.actual)).length / metrics.length) * 100 : 0;
  const metricsSource = metrics.length && metrics.every((item) => item.source === metrics[0].source) ? metrics[0].source : 'mixed';
  const metricsSourceLabel = metrics.length === 0
    ? isHistoricalEstimate ? 'ESTIMATE' : isDemoForecast ? 'DEMO' : 'MODEL'
    : metricsSource === 'historical_estimate' ? 'ESTIMATE' : metricsSource === 'demo' ? 'DEMO' : metricsSource === 'mixed' ? 'MIXED FORECAST' : 'MODEL';
  const errorPath = metrics.map((m, i) => `${i === 0 ? 'M' : 'L'} ${i * 18} ${34 - Math.min(29, Math.abs(m.predicted - m.actual) * 2)}`).join(' ');

  const filteredDams = dams.filter((item) => `${item.name} ${item.location} ${item.river}`.toLowerCase().includes(search.toLowerCase()));
  const isLoading = damsQuery.isLoading || (urlReady && historyQuery.isLoading);

  const setDam = (next: string) => {
    setDamId(next);
    setSelectedDam(next);
    const found = dams.find((item) => item.id === next);
    if (found) { setAsOf(found.lastRecorded.date); setDebouncedAsOf(found.lastRecorded.date); }
    setShowDamOptions(false);
    setSearch('');
    pendingStep.current = false;
  };
  const setWindow = (value: number) => setWindowDays(Math.max(120, Math.min(365, Math.round(value) || 120)));

  const forecastReady = !!predictedPoint;

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: PAGE_CSS }} />

      {/* The whole page is the reservoir: water height = current fill of the active range. */}
      {dam && (
        <WaterField
          dam={dam}
          currentFraction={currentFraction}
          predictedFraction={predictedFraction}
          currentLevel={currentLevel}
          predictedLevel={predictedLevel}
          horizon={horizon}
          showPredicted={showPredicted && forecastReady}
        />
      )}

      <section className="prediction-page">
        <div className="heading-row">
          <PageHeading
            label="OUTLOOK & BACKTESTING"
            title={<>Explore the<br /><span>days ahead.</span></>}
            description="Travel through observed readings, compare forecast horizons, and watch the reservoir fill or drain behind the page."
          />
          <div className="asof-status">
            <span className="status-pulse" />
            <span>As-of view</span>
            <strong>{asOf ? formatDate(asOf) : 'Loading…'}</strong>
          </div>
        </div>

        {damsQuery.isError && (
          <div className="glass-card error-card">
            <strong>Reservoir information could not be loaded.</strong>
            <p>Check your API connection and try again.</p>
            <button className="button button-ghost" onClick={() => damsQuery.refetch()}>Retry ↻</button>
          </div>
        )}

        {dam && (
          <>
            {/* ── controls ── */}
            <div className="deck-wrap">
              <Reveal>
                <section className="glass-card control-deck">
                  <div className="control" ref={selectorRef}>
                    <div className="control-label"><span>Reservoir</span></div>
                    <div className="dam-picker">
                      <Search size={16} />
                      <input
                        value={showDamOptions ? search : dam.name}
                        onChange={(e) => { setSearch(e.target.value); setShowDamOptions(true); }}
                        onFocus={() => { setSearch(''); setShowDamOptions(true); }}
                        onKeyDown={(e) => { if (e.key === 'ArrowDown') setShowDamOptions(true); if (e.key === 'Escape') setShowDamOptions(false); }}
                        aria-label="Search reservoirs"
                        placeholder="Search reservoir name"
                      />
                      <button type="button" onClick={() => { setSearch(''); setShowDamOptions((open) => !open); }} aria-expanded={showDamOptions} aria-label="Toggle reservoir list">⌄</button>
                      {showDamOptions && (
                        <div className="dam-options">
                          {filteredDams.map((item) => (
                            <button key={item.id} type="button" onClick={() => setDam(item.id)} className={dam.id === item.id ? 'selected' : ''}>
                              <span className="option-monogram">{item.name.slice(0, 1)}</span>
                              <span><strong>{item.name}</strong><small>{item.location} · {item.river}</small></span>
                              <span className="option-value">{fmt(item.lastRecorded.level)} {item.unit}</span>
                            </button>
                          ))}
                          {filteredDams.length === 0 && <div className="options-empty">No reservoirs found for “{search}”.</div>}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="control">
                  <div className="control-label"><span>Forecast horizon</span><strong>{forecastMethodLabel}</strong></div>
                    <div className="segmented-control" role="group" aria-label="Forecast horizon">
                      {HORIZONS.map((item) => (
                        <button key={item} className={item === horizon ? 'selected' : ''} onClick={() => setHorizon(item)}>
                          <strong>{item}</strong><span>{item === 1 ? 'day' : 'days'}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="control">
                    <div className="control-label">
                      <span>History window</span>
                      <strong>{windowDays} days</strong>
                    </div>
                    <input
                      className="range-slider"
                      type="range" min="120" max="365" step="1"
                      value={windowDays}
                      onChange={(e) => setWindow(Number(e.target.value))}
                      aria-label="History window from 120 to 365 days"
                    />
                  </div>
                </section>
              </Reveal>
            </div>

            {/* ── KPI strip ── */}
            <Reveal delay={0.04}>
              <div className="kpi-strip">
                <div className="glass-card kpi accent">
                  <span className="data-label">Waterline on {asOf ? formatDate(asOf) : '—'}</span>
                  <strong>{fmt(currentLevel)} <small>{dam.unit}</small></strong>
                  <span>{(currentFraction * 100).toFixed(0)}% of active range</span>
                </div>
                <div className="glass-card kpi">
                  <span className="data-label">{isHistoricalEstimate ? "Estimated" : isDemoForecast ? "Demo forecast" : "Predicted"} · {horizon}-day</span>
                  <strong>{forecastReady ? <>{fmt(predictedLevel)} <small>{dam.unit}</small></> : '—'}</strong>
                  <span>
                    {forecastReady
                      ? <span className={delta > 0 ? 'up' : delta < 0 ? 'down' : ''}>{delta > 0 ? '▲' : delta < 0 ? '▼' : '■'} {Math.abs(delta).toFixed(2)} {dam.unit} vs. as-of</span>
                      : 'Waiting for forecast'}
                  </span>
                </div>
                <div className="glass-card kpi">
                  <span className="data-label">Below full reservoir level</span>
                  <strong>{Math.max(0, dam.maxLevel - currentLevel).toFixed(2)} <small>{dam.unit}</small></strong>
                  <span>FRL {fmt(dam.maxLevel)} {dam.unit} · min {fmt(dam.minLevel)} {dam.unit}</span>
                </div>
                <div className="glass-card kpi kpi-toggle">
                  <span className="data-label">Water display</span>
                  <label className="tank-toggle">
                    <input type="checkbox" checked={showPredicted} onChange={(e) => setShowPredicted(e.target.checked)} />
                    <span className="switch" />
                    <span>Show predicted level</span>
                  </label>
                  <span>Dashed line on the water.</span>
                </div>
              </div>
            </Reveal>

            {/* ── chart + details ── */}
            <div className="main-grid">
              <Reveal delay={0.06}>
                <section className="glass-card chart-card">
                  <div className="card-head">
                    <div>
                      <Eyebrow>RESERVOIR LEVEL FORECAST</Eyebrow>
                      <h2>Observed & predicted level</h2>
                      <p>{dam.name} · {formatDate(debouncedAsOf || dam.lastRecorded.date)} base date · {horizon}-day horizon</p>
                    </div>
                    <div className="chart-legend">
                      <span><i className="legend-observed" /> Observed</span>
                      <span><i className="legend-forecast" /> Forecast</span>
                      <span><i className="legend-actual" /> Actual</span>
                    </div>
                  </div>

                  <div className="chart-wrap">
                    {historyQuery.isError || forecastQuery.isError ? (
                      <div className="chart-state">
                        <strong>Couldn’t load forecast data.</strong>
                        <span>Check the connection and retry.</span>
                        <button className="button button-ghost" onClick={() => { historyQuery.refetch(); forecastQuery.refetch(); }}>Retry ↻</button>
                      </div>
                    ) : isLoading || historyQuery.isFetching || forecastQuery.isFetching || !chartData.length ? (
                      <div className="chart-skeleton"><span /><span /><span /></div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 15, right: 13, bottom: 2, left: 2 }}>
                          <CartesianGrid stroke="rgba(166,217,228,.08)" vertical={false} />
                          <XAxis
                            dataKey="x" type="number" scale="time" domain={['dataMin', 'dataMax']}
                            tickFormatter={(v) => new Date(v).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                            tick={{ fill: '#8aa3ad', fontSize: 10 }} axisLine={{ stroke: 'rgba(166,217,228,.14)' }} tickLine={false} minTickGap={28}
                          />
                          {/* allowDataOverflow stops Recharts from widening the domain to include the stacked Area baseline (0). */}
                          <YAxis
                            width={60} tick={{ fill: '#8aa3ad', fontSize: 10 }} tickLine={false} axisLine={false}
                            domain={chartYDomain} allowDataOverflow tickCount={6} allowDecimals
                            tickFormatter={(v) => Number(v).toLocaleString('en-IN', { minimumFractionDigits: yDecimals, maximumFractionDigits: yDecimals })}
                          />
                          <Tooltip content={<ChartTooltip unit={dam.unit} />} cursor={{ stroke: 'rgba(150,235,244,.35)', strokeDasharray: '3 4' }} />
                          <ReferenceArea y1={dam.maxLevel} y2={dam.maxLevel + (dam.maxLevel - dam.minLevel) * 0.12} fill="#ff5660" fillOpacity={0.08} ifOverflow="hidden" />
                          <ReferenceLine y={dam.maxLevel} stroke="#ff8f8f" strokeDasharray="4 5" strokeOpacity={0.7} label={{ value: 'FRL', position: 'insideTopRight', fill: '#ffaaa9', fontSize: 9 }} />
                          <ReferenceLine x={dateToX(debouncedAsOf)} stroke="#b6edec" strokeDasharray="3 5" strokeOpacity={0.65} label={{ value: 'AS-OF', position: 'insideTopLeft', fill: '#b5d9dc', fontSize: 9 }} />
                          <Area type="monotone" dataKey="lower" stackId="confidence" stroke="none" fill="transparent" isAnimationActive={false} />
                          <Area type="monotone" dataKey="band" stackId="confidence" stroke="none" fill="#55d2df" fillOpacity={0.13} isAnimationActive={false} />
                          <Line type="monotone" dataKey="observed" name="Observed" stroke="#7ca4aa" strokeWidth={1.6} dot={false} connectNulls={false} isAnimationActive={false} />
                          <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#5de1ee" strokeWidth={2.2} strokeDasharray="6 4" dot={false} connectNulls isAnimationActive={false} />
                          <Line type="monotone" dataKey="actual" name="Actual after as-of" stroke="#d2a3ff" strokeWidth={2} dot={{ r: 2, fill: '#d2a3ff', strokeWidth: 0 }} connectNulls={false} isAnimationActive={false} />
                          <Brush dataKey="date" height={16} stroke="#55cbd6" travellerWidth={6} fill="rgba(3,14,19,.82)" tickFormatter={() => ''} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>

                  <div className="chart-foot">
                    <span><span className="risk-dot" /> Red zone begins above FRL</span>
                    <span>
                      {forecastQuery.isFetching ? 'Updating forecast' : isHistoricalEstimate ? 'Historical estimate only; uncertainty range is not calibrated' : isDemoForecast ? 'Illustrative demo forecast and range' : hasForecastBand ? 'Shaded band shows the confidence range, which widens with horizon' : 'Uncertainty band unavailable for this forecast'} <CircleHelp size={12} />
                    </span>
                  </div>
                </section>
              </Reveal>

              <aside className="details-column">
                <Reveal delay={0.08}>
                  <DamDetails dam={dam} currentLevel={currentLevel} currentDate={debouncedAsOf || dam.lastRecorded.date} />
                </Reveal>
              </aside>
            </div>

            {/* ── backtest + model comparison ── */}
            <div className="lower-grid">
              <Reveal delay={0.04}>
                <section className="glass-card backtest-card">
                  <div className="card-head">
                    <div>
                      <Eyebrow>{isHistoricalEstimate ? 'DATE TRAVEL & HISTORICAL ESTIMATE' : 'DATE TRAVEL & BACKTEST'}</Eyebrow>
                      <h2>Step through the record</h2>
                      <p>{isHistoricalEstimate ? 'Historical dates use a synthetic estimate, not the trained model; compare it with the observed readings that followed.' : 'Rebuild a forecast from an observation date and compare it with what followed.'}</p>
                    </div>
                    <button type="button" className="reset-session" onClick={() => { setMetrics([]); recordedSteps.current.clear(); pendingStep.current = false; }}>
                      <RotateCcw size={13} /> Reset session
                    </button>
                  </div>

                  <div className="time-controls">
                    <button className="step-button" type="button" onClick={() => step(-1)} disabled={asOf <= damMinDate}><ChevronLeft size={15} /><span>−1 day</span></button>
                    <label className="date-picker">
                      <CalendarDays size={15} />
                      <input aria-label="Choose as-of date" type="date" min={damMinDate} max={damLatestDate} value={asOf} onChange={(e) => { setAsOf(e.target.value); pendingStep.current = false; }} />
                    </label>
                    <button className="step-button" type="button" onClick={() => step(1)} disabled={asOf >= damLatestDate}><span>+1 day</span><ChevronRight size={15} /></button>
                    <button className="latest-button" type="button" onClick={() => { setAsOf(damLatestDate); setPlaying(false); pendingStep.current = false; }}>Latest</button>
                    <button className={`play-button ${playing ? 'playing' : ''}`} type="button" onClick={() => setPlaying((v) => !v)} disabled={asOf >= damLatestDate && !playing}>
                      {playing ? <Square size={13} fill="currentColor" /> : <Play size={13} fill="currentColor" />}
                      {playing ? 'Pause' : 'Play'}
                    </button>
                  </div>

                  <div className="play-speed">
                    <Clock3 size={13} />
                    <span>Step speed</span>
                    <input type="range" min="400" max="1800" step="100" value={playSpeed} onChange={(e) => setPlaySpeed(Number(e.target.value))} aria-label="Playback step speed" />
                    <strong>{(playSpeed / 1000).toFixed(1)}s / day</strong>
                    <span className="keyboard-note">← → to step</span>
                  </div>

                  <div className="result-grid">
                    <div className="result-current">
                      <span className="data-label">{formatDate(asOf || latestDate)} · {horizon}-day outlook</span>
                      <strong>{predictedPoint ? `${fmt(predictedPoint.level)} ${dam.unit}` : 'Waiting for forecast'}</strong>
                      <span>{actual ? `Observed ${fmt(actual.level)} ${dam.unit}` : 'Future actual not available at this date'}</span>
                    </div>
                    <div className="result-stat">
                      <span className="data-label">Absolute error</span>
                      <strong>{modelError !== undefined ? `${modelError.toFixed(2)} ${dam.unit}` : '—'}</strong>
                    </div>
                    <div className="result-stat">
                      <span className="data-label">Error %</span>
                      <strong>{modelError !== undefined && actual ? `${((modelError / Math.max(0.01, Math.abs(actual.level))) * 100).toFixed(2)}%` : '—'}</strong>
                    </div>
                    <div className="result-stat">
                      <span className="data-label">{isHistoricalEstimate ? 'Estimate vs. persistence' : isDemoForecast ? 'Demo vs. persistence' : 'Vs. persistence'}</span>
                      <strong>{modelError !== undefined && persistenceError !== undefined ? (modelError < persistenceError ? (isHistoricalEstimate ? 'Estimate wins' : isDemoForecast ? 'Demo wins' : 'Model wins') : 'Baseline wins') : '—'}</strong>
                    </div>
                  </div>

                  <div className="session-summary">
                    <div className="session-title">
                      <span className="data-label">Stepped days <b>{metrics.length}</b></span>
                      {metrics.length > 0 && (
                        <svg viewBox={`0 0 ${Math.max(1, (metrics.length - 1) * 18)} 36`} preserveAspectRatio="none">
                          <path d={errorPath} fill="none" stroke="#5cd9e3" strokeWidth="1.7" />
                        </svg>
                      )}
                    </div>
                    <div><span>RUNNING {metricsSourceLabel} MAE</span><strong>{metrics.length ? `${mae.toFixed(2)} ${dam.unit}` : '—'}</strong></div>
                    <div><span>RUNNING {metricsSourceLabel} RMSE</span><strong>{metrics.length ? `${rmse.toFixed(2)} ${dam.unit}` : '—'}</strong></div>
                    <div><span>{metricsSourceLabel} BEATS PERSISTENCE</span><strong>{metrics.length ? `${beats.toFixed(0)}%` : '—'}</strong></div>
                  </div>
                </section>
              </Reveal>

              <Reveal delay={0.08}>
                <section className="glass-card confidence-card">
                  <div className="card-head">
                    <div>
                      <Eyebrow>{isHistoricalEstimate ? 'HISTORICAL ESTIMATE & PERSISTENCE' : isDemoForecast ? 'DEMO FORECAST & PERSISTENCE' : 'MODEL CONFIDENCE & PERSISTENCE'}</Eyebrow>
                      <h2>{isHistoricalEstimate ? 'How does this estimate compare?' : isDemoForecast ? 'Illustrative demo forecast' : 'How does the model compare?'}</h2>
                      <p>{isHistoricalEstimate ? 'Historical estimates use a simple frontend heuristic; confidence is not calibrated.' : 'Persistence assumes the last observed level holds steady.'}</p>
                    </div>
                    <span className="tag">NAÏVE BASELINE</span>
                  </div>

                  <div className="horizon-comparison">
                    {HORIZONS.map((h, index) => {
                      // Only trust cached data that matches the current dam + as-of date,
                      // so a previous dam's numbers don't flash in these rows.
                      const raw = horizonQueries[index].data;
                      const f = raw && raw.damId === dam.id && raw.asOf === debouncedAsOf ? raw : undefined;
                      const confidence = f?.confidence[h] ?? 0;
                      const prediction = f?.predictions.at(-1)?.level;
                      const pers = f?.persistence[h];
                      const observed = f?.actuals?.at(-1)?.level;
                      const skill =
                        prediction !== undefined && pers !== undefined && observed !== undefined
                          ? ((Math.abs(pers - observed) - Math.abs(prediction - observed)) / Math.max(0.001, Math.abs(pers - observed))) * 100
                          : undefined;
                      return (
                        <button type="button" key={h} className={`horizon-row ${h === horizon ? 'active' : ''}`} onClick={() => setHorizon(h)}>
                          <span className="horizon-name">{h}<small>{h === 1 ? 'DAY' : 'DAYS'}</small></span>
                          <span className="confidence-meter">
                            <span className="data-label">Confidence</span>
                            <Meter value={confidence * 100} />
                            <strong>{confidence ? `${Math.round(confidence * 100)}%` : '—'}</strong>
                          </span>
                          <span className="value-pair">
                            <span>{f?.source === "historical_estimate" ? "Estimate" : f?.source === "demo" ? "Demo" : "Model"} <strong>{prediction !== undefined ? `${fmt(prediction)} ${dam.unit}` : '—'}</strong></span>
                            <span>Persistence <strong>{pers !== undefined ? `${fmt(pers)} ${dam.unit}` : '—'}</strong></span>
                          </span>
                          <span className={`skill-note ${skill !== undefined && skill > 0 ? 'positive' : ''}`}>
                            {skill !== undefined ? (skill > 0 ? `Beats by ${skill.toFixed(1)}%` : `${Math.abs(skill).toFixed(1)}% behind`) : 'No actual yet'}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              </Reveal>
            </div>

            <div className="footnote">
              <span>FORECAST NOTE</span>
              <p>
                {isHistoricalEstimate
                  ? 'This backdated forecast is a synthetic estimate, not a model backtest. Its confidence range is suppressed because it is not calibrated.'
                  : isDemoForecast
                    ? 'Demo data and forecasts are illustrative and do not come from the trained reservoir model.'
                    : 'The backend applies the saved persistence, recent-trend, or Delta-LSTM strategy for each horizon.'}
              </p>
            </div>
          </>
        )}
      </section>
    </>
  );
}

/* ───────────────────────── water field ───────────────────────── */

function WaterField({
  dam, currentFraction, predictedFraction, currentLevel, predictedLevel, horizon, showPredicted,
}: {
  dam: Dam;
  currentFraction: number;
  predictedFraction: number;
  currentLevel: number;
  predictedLevel: number;
  horizon: Horizon;
  showPredicted: boolean;
}) {
  // Keep a thin sliver of water visible even at the dead-pool level.
  const level = Math.max(0.015, currentFraction) * 100;
  const predicted = Math.max(0.015, predictedFraction) * 100;
  const ticks = [0, 25, 50, 75, 100];
  // The water rises or falls to the predicted level when "Show predicted level" is on.
  const surface = showPredicted ? predicted : level;

  return (
    <div className="water-field" aria-hidden="true">
      <div className="fx-layer"><div className="fx-sun" /></div>

      <div className="water-body" style={{ height: `${surface}%` }}>
        <svg className="water-wave back" viewBox={`0 0 ${WAVE_W} ${WAVE_H}`} preserveAspectRatio="none"><path d={WAVE_BACK} /></svg>
        <svg className="water-wave front" viewBox={`0 0 ${WAVE_W} ${WAVE_H}`} preserveAspectRatio="none"><path d={WAVE_FRONT} /></svg>
        <div className="water-surface-glow" />

        {/* effects clipped to the water, so they scale with the level */}
        <div className="water-fx">
          <div className="fx-caustics a" />
          <div className="fx-caustics b" />
          {FX.fish.map((f, i) => (
            <div key={i} className={`fx-fish ${f.rev ? 'rev' : ''}`}
              style={{ top: f.top, '--d': f.dur, '--dl': f.delay, '--fs': f.s } as CSSProperties}>
              <svg viewBox="0 0 42 16"><path d={FISH_PATH} /></svg>
            </div>
          ))}
          {FX.bubbles.map((b, i) => (
            <span key={i} className="fx-bubble"
              style={{ '--l': b.left, '--s': b.s, '--d': b.dur, '--dl': b.delay, '--sw': b.sway } as CSSProperties}>
              <i />
            </span>
          ))}
        </div>
      </div>

      {/* light rays, marine snow and vignette */}
      <div className="fx-layer fx-rays">
        {FX.rays.map((r, i) => (
          <div key={i} className="fx-ray"
            style={{ left: r.left, width: r.w, '--r': r.rot, '--d': r.dur, '--dl': r.delay, '--o': r.o } as CSSProperties} />
        ))}
      </div>
      <div className="fx-layer fx-snow">
        {FX.snow.map((s, i) => (
          <span key={i} style={{ '--l': s.left, '--t': s.top, '--s': s.s, '--d': s.dur, '--dl': s.delay, '--dx': s.dx } as CSSProperties} />
        ))}
      </div>
      <div className="fx-layer fx-vignette" />

      {showPredicted && (
        <>
          <div className="predicted-band" style={{ bottom: `${Math.min(level, predicted)}%`, height: `${Math.abs(predicted - level)}%` }} />
          <div className="predicted-line" style={{ bottom: `${predicted}%` }}>
            <span>{horizon}-DAY FORECAST · {predictedLevel.toLocaleString('en-IN')} {dam.unit}</span>
          </div>
        </>
      )}

      <div className="water-gauge">
        {ticks.map((t) => {
          const lvl = dam.minLevel + ((dam.maxLevel - dam.minLevel) * t) / 100;
          return <div key={t} className="tick" style={{ bottom: `${t}%` }}>{t === 0 || t === 100 ? lvl.toLocaleString('en-IN') : `${t}%`}</div>;
        })}
        <div className="marker" style={{ bottom: `${level}%` }}><b>{currentLevel.toLocaleString('en-IN')} {dam.unit}</b></div>
      </div>
    </div>
  );
}

/* ───────────────────────── sub-components ───────────────────────── */

function useForecastQuery(dam: Dam | undefined, history: LevelPoint[], asOf: string, horizon: Horizon, enabled: boolean) {
  return useQuery({
    queryKey: ['forecast', dam?.id, asOf, horizon],
    queryFn: ({ signal }) => fetchForecast(dam!, history, asOf, horizon, signal),
    enabled: enabled && !!dam && !!asOf,
    staleTime: 0,
  });
}

function ChartTooltip({ active, payload, label, unit }: {
  active?: boolean;
  payload?: Array<{ dataKey?: string; value?: number; payload?: { date: string } }>;
  label?: number;
  unit: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload.find((p) => p.value !== undefined)?.payload;
  const date = row?.date ?? (label ? new Date(label).toISOString().slice(0, 10) : '');
  const values = payload.filter((p) => p.value !== undefined && p.dataKey && ['observed', 'forecast', 'actual'].includes(p.dataKey));
  return (
    <div className="chart-tooltip">
      <span>{date ? formatDate(date) : '—'}</span>
      {values.map((p) => (
        <div key={p.dataKey}>
          <i className={`tooltip-${p.dataKey}`} />
          <span>{p.dataKey === 'actual' ? 'Actual' : p.dataKey === 'observed' ? 'Observed' : 'Predicted'}</span>
          <strong>{Number(p.value).toLocaleString('en-IN')} {unit}</strong>
        </div>
      ))}
    </div>
  );
}

function DamDetails({ dam, currentLevel, currentDate }: { dam: Dam; currentLevel: number; currentDate: string }) {
  const lat = Math.max(9, Math.min(20, dam.lat));
  const lng = Math.max(72, Math.min(79, dam.lng));
  const x = 13 + (lng - 72) * 11.3;
  const y = 87 - (lat - 9) * 7.1;

  const activeFraction = clamp01((currentLevel - dam.minLevel) / (dam.maxLevel - dam.minLevel));
  const asOfStorage =
    currentDate >= dam.lastRecorded.date
      ? dam.currentStorage ?? dam.deadStorage + activeFraction * dam.liveStorage
      : dam.deadStorage + activeFraction * dam.liveStorage;

  const fields: [string, string][] = [
    ['RESERVOIR', dam.reservoir],
    ['RIVER', dam.river],
    ['DISTRICT', dam.district],
    ['STATE', dam.state],
    ['CONSTRUCTED', dam.yearBuilt ? String(dam.yearBuilt) : 'Not listed'],
    ['PURPOSE', dam.purpose],
    ['GROSS STORAGE', `${fmt(dam.grossStorage)} TMC`],
    ['LIVE STORAGE CAPACITY', `${fmt(dam.liveStorage)} TMC`],
    ['AS-OF LIVE STORAGE · EST.', `${asOfStorage.toFixed(2)} TMC`],
    ['DEAD STORAGE', `${fmt(dam.deadStorage)} TMC`],
    ['CATCHMENT AREA', `${fmt(dam.catchmentArea)} km²`],
    ['FULL RESERVOIR LEVEL', `${fmt(dam.maxLevel)} ${dam.unit}`],
    ['MIN / DEAD-POOL LEVEL', `${fmt(dam.minLevel)} ${dam.unit}`],
  ];

  return (
    <section className="glass-card details-card">
      <div className="details-title">
        <Eyebrow>DAM FIELD NOTES</Eyebrow>
        <span className="details-emblem"><Waves size={18} /></span>
      </div>
      <h2>{dam.name}</h2>
      <p className="details-location"><MapPin size={13} />{dam.location}</p>

      <div className="details-asof">
        <span>LEVEL ON {formatDate(currentDate).toUpperCase()}</span>
        <strong>{fmt(currentLevel)} <small>{dam.unit}</small></strong>
      </div>

      <div className="detail-grid">
        {fields.map(([label, value]) => (
          <div key={label}><span>{label}</span><strong>{value}</strong></div>
        ))}
      </div>

      <div className="places-section">
        <span className="data-label">Places around the reservoir</span>
        <div className="chip-row">{dam.nearbyPlaces.map((p) => <span className="chip" key={p}>{p}</span>)}</div>
      </div>
      <div className="places-section">
        <span className="data-label">Overflow-affected regions</span>
        <div className="chip-row">{dam.affectedRegions.map((p) => <span className="chip chip-alert" key={p}>{p}</span>)}</div>
      </div>

      <div className="location-map">
        <div className="map-watermark">KARNATAKA</div>
        <svg viewBox="0 0 100 100" aria-label={`Schematic location map for ${dam.name}`}>
          <path className="map-outline" d="M35 5 48 9 54 17 67 20 72 29 68 37 78 44 72 55 77 65 70 74 72 84 61 91 50 84 39 94 28 85 24 73 17 66 22 55 14 45 21 34 19 24 27 16 28 9Z" />
          <path className="map-river" d="M29 79c10-12 8-18 20-23s15-10 19-22" />
          <circle cx={x} cy={y} r="5.5" className="map-ping" />
          <circle cx={x} cy={y} r="2" className="map-pin" />
        </svg>
        <div className="map-label">
          <span>{dam.district}</span>
          <strong>{dam.lat.toFixed(3)}° N <i />{dam.lng.toFixed(3)}° E</strong>
        </div>
      </div>

      <div className="details-footer"><span>As-of observation</span><strong>{formatDate(currentDate)}</strong></div>
    </section>
  );
}
