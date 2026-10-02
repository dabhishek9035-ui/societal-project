'use client';

import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { ArrowDownRight, ArrowRight, Code2, Waves, Droplets, Activity } from 'lucide-react';
import { fetchDams } from '@/data/api';
import { projectInfo, GITHUB_URL } from '@/config';
import { Dam } from '@/data/types';
import { Eyebrow, Reveal } from '@/components/Ui';

export default function HomePage() {
  const { data: dams = [] } = useQuery({ queryKey: ['dams'], queryFn: ({ signal }) => fetchDams(signal) });
  return <>
    <section className="home-hero">
      <div className="hero-copy">
        <Eyebrow>{projectInfo.eyebrow}</Eyebrow>
        <h1>Water moves.<br /><span>We look ahead.</span></h1>
        <p className="hero-tagline">{projectInfo.tagline}</p>
        <p className="hero-highlight">{projectInfo.highlight}</p>
        <div className="hero-actions"><Link href="/dams" className="button button-primary">Explore reservoirs <ArrowRight size={16}/></Link><Link href="/predictions" className="button button-ghost">Run a forecast <span className="button-arrow">↗</span></Link></div>
        <div className="hero-footnote"><span className="footnote-rule"/> DAILY SIGNALS, LONGER HORIZONS</div>
      </div>
      <div className="hero-orbit" aria-hidden="true"><div className="orbit-ring ring-one"/><div className="orbit-ring ring-two"/><div className="orbit-ring ring-three"/><div className="orbit-core"><span/><i/></div><div className="orbit-label orbit-top">LIVE WATERLINE <b>↗</b></div><div className="orbit-label orbit-bottom">KARNATAKA <b>10 / 10</b></div><div className="orbit-coord">14° 31′ N<br/>75° 43′ E</div></div>
      <a href="#overview" className="scroll-cue"><span className="scroll-mouse"><i/></span><span>SCROLL TO EXPLORE</span><ArrowDownRight size={14}/></a>
    </section>
    <section className="home-overview" id="overview">
      <div className="section-topline"><Eyebrow>A FIELD GUIDE TO THE WATERLINE</Eyebrow><span>01 — 03</span></div>
      <div className="home-card-grid">
        <Reveal><article className="glass-card feature-card description-card"><div className="feature-icon"><Waves size={20}/></div><span className="card-kicker">01 / THE PROJECT</span><h2>Every reservoir<br/>has a story.</h2><p>{projectInfo.description}</p><div className="mini-metrics"><div><strong>10</strong><span>reservoirs</span></div><div><strong>30<span>d</span></strong><span>forecast range</span></div><div><strong>4</strong><span>time horizons</span></div></div><Link href="/predictions" className="card-arrow">Explore the outlook <span>↗</span></Link></article></Reveal>
        <Reveal delay={.08}><article className="glass-card feature-card github-card"><div className="feature-icon"><Code2 size={20}/></div><span className="card-kicker">02 / OPEN PROJECT</span><h2>Built in the<br/>open water.</h2><p>{projectInfo.repositoryBlurb}</p><div className="repo-box"><span className="repo-branch"><i/> MAIN BRANCH</span><strong>{projectInfo.repository}</strong><span className="repo-sub">HYDROLOGY · FORECASTING · DATA</span></div><a href={GITHUB_URL} target="_blank" rel="noreferrer" className="button button-ghost button-wide">View on GitHub <span className="button-arrow">↗</span></a></article></Reveal>
        <Reveal delay={.16}><article className="glass-card feature-card list-card"><div className="feature-icon"><Droplets size={20}/></div><span className="card-kicker">03 / THE NETWORK</span><h2>Across the<br/>river basins.</h2><p>Follow the latest reported level at each monitored reservoir.</p><div className="compact-dam-list">{dams.map((dam:Dam,i)=><Link href={`/dams#${dam.id}`} key={dam.id}><span className="list-index">{String(i+1).padStart(2,'0')}</span><strong>{dam.name.replace(' Dam','')}</strong><span className="list-level">{dam.lastRecorded.level.toLocaleString('en-IN')} {dam.unit}</span><span className="list-arrow">↗</span></Link>)}</div><Link href="/dams" className="card-arrow">View all {dams.length || 10} reservoirs <span>↗</span></Link></article></Reveal>
      </div>
      <div className="home-data-strip"><div><Activity size={15}/><span>{projectInfo.dataSummary}</span></div><span>OBSERVE · COMPARE · PLAN</span></div>
    </section>
    <footer className="site-footer"><Link href="/" className="footer-brand">JALADRISHTI <span>WATER INTELLIGENCE</span></Link><p>Made for a more informed view of Karnataka’s water.</p><span className="footer-mark">INDIA · 2026</span></footer>
  </>;
}
