'use client';

import { motion, useReducedMotion } from 'framer-motion';
import Link from 'next/link';
import type { Dam } from '@/data/types';

export function Reveal({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return <motion.div className={className} initial={reduce ? false : { opacity: 0, y: 22 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .14 }} transition={{ duration: reduce ? .01 : .6, delay, ease: [.2,.75,.25,1] }}>{children}</motion.div>;
}

export function Eyebrow({ children }: { children: React.ReactNode }) { return <div className="eyebrow"><span className="eyebrow-mark" />{children}</div>; }

export function PageHeading({ label, title, description, count }: { label: string; title: React.ReactNode; description: string; count?: string }) {
  return <div className="page-heading"><div><Eyebrow>{label}</Eyebrow><h1>{title}</h1><p>{description}</p></div>{count && <div className="heading-count"><strong>{count.split(' ')[0]}</strong><span>{count.split(' ').slice(1).join(' ')}</span></div>}</div>;
}

export function Meter({ value, max = 100, label }: { value: number; max?: number; label?: string }) {
  const percent = Math.max(0, Math.min(100, value / max * 100));
  return <div className="meter-wrap">{label && <div className="meter-label"><span>{label}</span><span>{percent.toFixed(0)}%</span></div>}<div className="meter"><span style={{ width: `${percent}%` }} /></div></div>;
}

export function DamCard({ dam, index = 0 }: { dam: Dam; index?: number }) {
  const full = Math.max(0, Math.min(100, (dam.lastRecorded.level - dam.minLevel) / (dam.maxLevel - dam.minLevel) * 100));
  return <Reveal delay={Math.min(index % 4, 3) * .07}><article className="glass-card dam-card" id={dam.id}>
    <div className="dam-card-top"><span className="dam-index">{String(index + 1).padStart(2,'0')}</span><span className="live-tag"><i /> READING</span></div>
    <h2>{dam.name}</h2><p className="dam-location"><MapPin />{dam.location} <span>·</span> {dam.river}</p>
    <div className="dam-card-grid"><section><span className="data-label">PREVIOUS LEVEL</span><strong>{dam.lastRecorded.level.toLocaleString('en-IN')} <small>{dam.unit}</small></strong><span className="data-sub">Recorded {formatDate(dam.lastRecorded.date)}</span></section><section className="fill-stat"><span className="data-label">RESERVOIR FILL</span><strong>{full.toFixed(0)}<small>%</small></strong><Meter value={full} /></section></div>
    <div className="regions-block"><span className="data-label">REGIONS AFFECTED IF OVERFLOW</span><div className="chip-row">{dam.affectedRegions.map((region)=><span className="chip" key={region}>{region}</span>)}</div></div>
    <div className="dam-card-bottom"><span className="capacity-note">FRL {dam.maxLevel.toLocaleString('en-IN')} {dam.unit}</span><Link className="text-link predict-link" href={`/predictions?dam=${dam.id}`}>Predict <span>↗</span></Link></div>
  </article></Reveal>;
}

function MapPin(){return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none"><path d="M19 10c0 5-7 11-7 11S5 15 5 10a7 7 0 1 1 14 0Z" stroke="currentColor" strokeWidth="1.5"/><circle cx="12" cy="10" r="2.2" stroke="currentColor" strokeWidth="1.5"/></svg>}
export function formatDate(value: string) { return new Date(`${value}T12:00:00`).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }); }
